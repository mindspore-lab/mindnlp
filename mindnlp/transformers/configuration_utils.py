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
# pylint: disable=C0103
# pylint: disable=W1203
# pylint: disable=W0613
# pylint: disable=C0415
"""
Pretrained config.
"""

import copy
import json
import os
import re
from typing import Optional, Tuple, Dict, Union, Any, List
from mindspore import jit_class

from mindnlp.utils import logging, is_mindspore_available
from mindnlp.configs import CONFIG_NAME
from mindnlp.utils.download import is_remote_url, download_url, cached_file

logger = logging.get_logger(__name__)

@jit_class
class PretrainedConfig:
    """
    Abstract class for Pretrained models config.
    """
    is_composition = False
    # Add for handle attribute_map
    attribute_map: Dict[str, str] = {}

    def __init__(self, **kwargs):
        self.ms_dtype = kwargs.pop("ms_dtype", None)
        if 'torch_dtype' in kwargs:
            self.ms_dtype = kwargs.pop("torch_dtype", None)
        self.return_dict = kwargs.pop("return_dict", True)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)

        self.pruned_heads = kwargs.pop("pruned_heads", {})
        self.tie_word_embeddings = kwargs.pop(
            "tie_word_embeddings", True
        )  # Whether input and output word embeddings should be tied for all MLM, LM and Seq2Seq models.

        # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.is_decoder = kwargs.pop("is_decoder", False)
        self.cross_attention_hidden_size = kwargs.pop("cross_attention_hidden_size", None)
        self.add_cross_attention = kwargs.pop("add_cross_attention", False)
        self.tie_encoder_decoder = kwargs.pop("tie_encoder_decoder", False)

       # Parameters for sequence generation
        self.max_length = kwargs.pop("max_length", 20)
        self.min_length = kwargs.pop("min_length", 0)
        self.do_sample = kwargs.pop("do_sample", False)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.typical_p = kwargs.pop("typical_p", 1.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)
        self.output_scores = kwargs.pop("output_scores", False)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)
        self.exponential_decay_length_penalty = kwargs.pop("exponential_decay_length_penalty", None)
        self.suppress_tokens = kwargs.pop("suppress_tokens", None)
        self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)

        # Fine-tuning task arguments
        self.architectures = kwargs.pop("architectures", None)
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.id2label = kwargs.pop("id2label", None)
        self.label2id = kwargs.pop("label2id", None)
        if self.label2id is not None and not isinstance(self.label2id, dict):
            raise ValueError("Argument label2id should be a dictionary.")
        if self.id2label is not None:
            if not isinstance(self.id2label, dict):
                raise ValueError("Argument id2label should be a dictionary.")
            num_labels = kwargs.pop("num_labels", None)

            if num_labels is not None and len(self.id2label) != num_labels:
                logger.warning(
                    f"You passed along `num_labels={num_labels}` with an incompatible id to label map: "
                    f"{self.id2label}. The number of labels wil be overwritten to {self.num_labels}."
                )
            self.id2label = {int(key): value for key, value in self.id2label.items()}
            # Keys are always strings in JSON so convert ids to int here.
        else:
            self.num_labels = kwargs.pop("num_labels", 2)

        if self.ms_dtype is not None and isinstance(self.ms_dtype, str):
            if is_mindspore_available():
                import mindspore

                self.ms_dtype = getattr(mindspore, self.ms_dtype)

        # Tokenizer arguments TODO: eventually tokenizer and models should share the same config
        self.tokenizer_class = kwargs.pop("tokenizer_class", None)
        self.prefix = kwargs.pop("prefix", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.sep_token_id = kwargs.pop("sep_token_id", None)

        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # task specific arguments
        self.task_specific_params = kwargs.pop("task_specific_params", None)

        # regression / multi-label classification
        self.problem_type = kwargs.pop("problem_type", None)
        allowed_problem_types = ("regression", "single_label_classification", "multi_label_classification")
        if self.problem_type is not None and self.problem_type not in allowed_problem_types:
            raise ValueError(
                f"The config parameter `problem_type` was not understood: received {self.problem_type} "
                "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
            )

        # Name or path to the pretrained checkpoint
        self._name_or_path = str(kwargs.pop("name_or_path", ""))

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    @property
    def name_or_path(self) -> str:
        """get name_or_path"""
        return getattr(self, "_name_or_path", None)

    @name_or_path.setter
    def name_or_path(self, value):
        """set name_or_path"""
        self._name_or_path = str(value)  # Make sure that name_or_path is a string (for JSON encoding)

    @classmethod
    def from_json(cls, file_path):
        """load config from json."""
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        config_map = json.loads(text)
        config = cls()
        for key, value in config_map.items():
            setattr(config, key, value)
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `Config` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        dict_obj = json.loads(text)
        return cls(**dict_obj)

    @classmethod
    def load(cls, pretrained_model_name_or_path):
        """load config."""
        return cls.from_pretrained(pretrained_model_name_or_path)

    @property
    def use_return_dict(self) -> bool:
        """
        `bool`: Whether or not return [`~utils.ModelOutput`] instead of tuples.
        """
        # If torchscript is set, force `return_dict=False` to avoid jit errors
        return self.return_dict

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "PretrainedConfig":
        """
        Constructs a `Config` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be retrieved
                from a pre-trained checkpoint by leveraging the :func:`~transformers.PretrainedConfig.get_config_dict`
                method.
            kwargs (:obj:`Dict[str, any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`PretrainedConfig`: An instance of a configuration object
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        # Update config with kwargs if needed
        if "num_labels" in kwargs and "id2label" in kwargs:
            num_labels = kwargs["num_labels"]
            id2label = kwargs["id2label"] if kwargs["id2label"] is not None else []
            if len(id2label) != num_labels:
                raise ValueError(
                    f"You passed along `num_labels={num_labels }` with an incompatible id to label map: "
                    f"{kwargs['id2label']}. Since those arguments are inconsistent with each other, you should remove "
                    "one of them.")

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", str(config))
        if return_unused_kwargs:
            return config, kwargs
        return config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        **kwargs,
    ) -> "PretrainedConfig":
        r"""
        Instantiate a [`PretrainedConfig`] (or a derived class) from a pretrained model configuration.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a configuration file saved using the
                  [`~PretrainedConfig.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved configuration JSON *file*, e.g., `./my_model_directory/configuration.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
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
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from this pretrained model.

        Examples:

        ```python
        # We can't instantiate directly the base class *PretrainedConfig* so let's show the examples on a
        # derived class: BertConfig
        config = BertConfig.from_pretrained(
            "bert-base-uncased"
        )  # Download configuration from huggingface.co and cache.
        config = BertConfig.from_pretrained(
            "./test/saved_model/"
        )  # E.g. config (or model) was saved using *save_pretrained('./test/saved_model/')*
        config = BertConfig.from_pretrained("./test/saved_model/my_configuration.json")
        config = BertConfig.from_pretrained("bert-base-uncased", output_attentions=True, foo=False)
        assert config.output_attentions == True
        config, unused_kwargs = BertConfig.from_pretrained(
            "bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
        )
        assert config.output_attentions == True
        assert unused_kwargs == {"foo": False}
        ```"""
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        [`PretrainedConfig`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.

        """
        original_kwargs = copy.deepcopy(kwargs)
        # Get config dict associated with the base config file
        config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)

        # That config file may point us toward another config file to use.
        if "configuration_files" in config_dict:
            configuration_file = get_configuration_file(config_dict["configuration_files"])
            config_dict, kwargs = cls._get_config_dict(
                pretrained_model_name_or_path, _configuration_file=configuration_file, **original_kwargs
            )

        return config_dict, kwargs

    @classmethod
    def _get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used
        for instantiating a Config using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (:obj:`string`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary that will be used to instantiate the configuration object.

        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        subfolder = kwargs.pop("subfolder", "")
        token = kwargs.pop('token', None)
        revision = kwargs.pop('revision', 'main')

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            # Special case when pretrained_model_name_or_path is a local file
            resolved_config_file = pretrained_model_name_or_path
            is_local = True

        elif is_remote_url(pretrained_model_name_or_path):
            configuration_file = pretrained_model_name_or_path
            resolved_config_file = download_url(pretrained_model_name_or_path)

        else:
            configuration_file = kwargs.pop("_configuration_file", CONFIG_NAME)

            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    configuration_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    revision=revision,
                    token=token,
                    subfolder=subfolder,
                )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception as exc:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it"
                    ", make sure you don't have a local directory with the same"
                    f" name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory"
                    f" containing a {configuration_file} file"
                ) from exc

        try:
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file."
            ) from exc

        if is_local:
            logger.info(f"loading configuration file {resolved_config_file}")
        else:
            logger.info(f"loading configuration file {configuration_file} from cache at {resolved_config_file}")

        return config_dict, kwargs

    @classmethod
    def _dict_from_json_file(cls, json_file: str):
        """_dict_from_json_file"""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def dict_ms_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("ms_dtype", None) is not None and not isinstance(d["ms_dtype"], str):
            d["ms_dtype"] = str(d["ms_dtype"]).lower()
        for value in d.values():
            if isinstance(value, dict):
                self.dict_ms_dtype_to_str(value)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        if "_auto_class" in output:
            del output["_auto_class"]
        if "_commit_hash" in output:
            del output["_commit_hash"]
        if "_attn_implementation_internal" in output:
            del output["_attn_implementation_internal"]

        for key, value in output.items():
            # Deal with nested configs like CLIP
            if isinstance(value, PretrainedConfig):
                value = value.to_dict()

            output[key] = value

        if hasattr(self, "quantization_config"):
            output["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )

            # pop the `_pre_quantization_dtype` as torch.dtypes are not serializable.
            _ = output.pop("_pre_quantization_dtype", None)

        self.dict_ms_dtype_to_str(output)

        return output

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = PretrainedConfig().to_dict()

        # get class specific config dict
        class_config_dict = self.__class__().to_dict() if not self.is_composition else {}

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if (
                isinstance(getattr(self, key, None), PretrainedConfig)
                and key in class_config_dict
                and isinstance(class_config_dict[key], dict)
            ):
                # For nested configs we need to clean the diff recursively
                diff = recursive_diff_dict(value, class_config_dict[key], config_obj=getattr(self, key, None))
                if "model_type" in value:
                    # Needs to be set even if it's not in the diff
                    diff["model_type"] = value["model_type"]
                if len(diff) > 0:
                    serializable_config_dict[key] = diff
            elif (
                key not in default_config_dict
                or value != default_config_dict[key]
                or (key in class_config_dict and value != class_config_dict[key])
            ):
                serializable_config_dict[key] = value

        return serializable_config_dict

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_file(self, save_path):
        """Serializes this instance to a JSON file."""
        output_dict = self.to_dict()
        with open(os.path.join(save_path, 'config.json'), encoding='utf-8') as f:
            json.dump(output_dict, f, sort_keys=True, indent=2)

    def update(self, config_dict: Dict[str, Any]):
        """
        Updates attributes of this class with attributes from `config_dict`.

        Args:
            config_dict (`Dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    @property
    def num_labels(self) -> int:
        """
        `int`: The number of labels for classification models.
        """
        return len(self.id2label)

    @num_labels.setter
    def num_labels(self, num_labels: int):
        if not hasattr(self, "id2label") or self.id2label is None or len(self.id2label) != num_labels:
            self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
            self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))


class EncoderDecoderConfig(PretrainedConfig):
    r"""
    [`EncoderDecoderConfig`] is the configuration class to store the configuration of a [`EncoderDecoderModel`]. It is
    used to instantiate an Encoder Decoder model according to the specified arguments, defining the encoder and decoder
    configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the encoder config.
                - **decoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the decoder config.

    Examples:

    ```python
    >>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

    >>> # Initializing a BERT bert-base-uncased style configuration
    >>> config_encoder = BertConfig()
    >>> config_decoder = BertConfig()

    >>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    >>> # Initializing a Bert2Bert model (with random weights) from the bert-base-uncased style configurations
    >>> model = EncoderDecoderModel(config=config)

    >>> # Accessing the model configuration
    >>> config_encoder = model.config.encoder
    >>> config_decoder = model.config.decoder
    >>> # set decoder config to causal lm
    >>> config_decoder.is_decoder = True
    >>> config_decoder.add_cross_attention = True

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("my-model")

    >>> # loading model and config from pretrained folder
    >>> encoder_decoder_config = EncoderDecoderConfig.from_pretrained("my-model")
    >>> model = EncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
    ```"""
    model_type = "encoder-decoder"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            "encoder" in kwargs and "decoder" in kwargs
        ), "Config has to be initialized with encoder and decoder config"
        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")
        decoder_config = kwargs.pop("decoder")
        decoder_model_type = decoder_config.pop("model_type")

        from mindnlp.transformers.models.auto.configuration_auto import AutoConfig

        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.is_encoder_decoder = True

    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`EncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model configuration and
        decoder model configuration.

        Returns:
            [`EncoderDecoderConfig`]: An instance of a configuration object
        """
        logger.info("Set `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)

def get_configuration_file(configuration_files: List[str]) -> str:
    """
    Get the configuration file to use for this version of transformers.

    Args:
        configuration_files (`List[str]`): The list of available configuration files.

    Returns:
        `str`: The configuration file to use.
    """
    configuration_files_map = {}
    for file_name in configuration_files:
        search = re.compile(r"config\.(.*)\.json").search(file_name)
        if search is not None:
            v = search.groups()[0]
            configuration_files_map[v] = file_name

    # Defaults to FULL_CONFIGURATION_FILE and then try to look at some newer versions.
    configuration_file = CONFIG_NAME

    return configuration_file


def recursive_diff_dict(dict_a, dict_b, config_obj=None):
    """
    Helper function to recursively take the diff between two nested dictionaries. The resulting diff only contains the
    values from `dict_a` that are different from values in `dict_b`.
    """
    diff = {}
    default = config_obj.__class__().to_dict() if config_obj is not None else {}
    for key, value in dict_a.items():
        obj_value = getattr(config_obj, str(key), None)
        if isinstance(obj_value, PretrainedConfig) and key in dict_b and isinstance(dict_b[key], dict):
            diff_value = recursive_diff_dict(value, dict_b[key], config_obj=obj_value)
            if len(diff_value) > 0:
                diff[key] = diff_value
        elif key not in dict_b or value != dict_b[key] or key not in default or value != default[key]:
            diff[key] = value
    return diff
