# Copyright 2022 Huawei Technologies Co., Ltd
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
Abstract class for Pretrained models.
"""
# pylint: disable=E1128

import copy
import json
import os
import logging
from typing import Union, Optional, Tuple, Dict
import mindspore
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import nn, ops

from ...utils.download import cached_path

logger = logging.getLogger(__name__)

class PretrainedConfig:
    """
    Abstract class for Pretrained models config.
    """
    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.is_decoder = kwargs.pop("is_decoder", False)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.add_cross_attention = kwargs.pop("add_cross_attention", False)
        self.tie_word_embeddings = kwargs.pop(
            "tie_word_embeddings", True
        )  # Whether input and output word embeddings should be tied for all MLM, LM and Seq2Seq models.
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)
        self.return_dict = kwargs.pop("return_dict", False)
        self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)
        self.pruned_heads = kwargs.pop("pruned_heads", {})

        self.problem_type = kwargs.pop("problem_type", None)
        allowed_problem_types = ("regression", "single_label_classification", "multi_label_classification")
        if self.problem_type is not None and self.problem_type not in allowed_problem_types:
            raise ValueError(
                f"The config parameter `problem_type` was not understood: received {self.problem_type} "
                "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
            )

    pretrained_config_archive_map: Dict[str, str] = {}

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
        if os.path.exists(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            raise ValueError(
                f"unable to parse {pretrained_model_name_or_path} as a local path or model name")

        config = cls.from_json(config_file)

        return config

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
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> "PretrainedConfig":
        """from_pretrained"""
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: str, pretrained_config_archive_map: Optional[Dict] = None, **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used
        for instantiating a Config using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (:obj:`string`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            pretrained_config_archive_map: (:obj:`Dict[str, str]`, `optional`) Dict:
                A map of `shortcut names` to `url`. By default, will use the current class attribute.

        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary that will be used to instantiate the configuration object.

        """
        cache_dir = kwargs.pop("cache_dir", None)
        kwargs.pop("force_download", False)
        kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        kwargs.pop("local_files_only", False)

        if pretrained_config_archive_map is None:
            pretrained_config_archive_map = cls.pretrained_config_archive_map

        if pretrained_model_name_or_path in pretrained_config_archive_map:
            config_file = pretrained_config_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        elif os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path

        try:
            # Load from URL or cache if already cached
            resolved_config_file = str(cached_path(
                config_file,
                cache_dir=cache_dir,
                proxies=proxies,
            ))
            # Load config dict
            if resolved_config_file is None:
                raise EnvironmentError
            config_dict = cls._dict_from_json_file(resolved_config_file)

        except EnvironmentError as exc:
            if pretrained_model_name_or_path in pretrained_config_archive_map:
                msg = f"Couldn't reach server at '{config_file}' to download pretrained model configuration file."
            else:
                msg = (
                    f"Can't load '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' "
                    f"is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                    f"- or '{pretrained_model_name_or_path}' "
                    f"is the correct path to a directory containing a config.json file\n\n"
                )
            raise EnvironmentError(msg) from exc

        except json.JSONDecodeError as exc:
            msg = (
                f"Couldn't reach server at '{config_file}' to download configuration file or "
                f"configuration file is not a valid JSON file. "
                f"Please check network or file content here: {resolved_config_file}."
            )
            raise EnvironmentError(msg) from exc

        if resolved_config_file == config_file:
            logger.info("loading configuration file %s", config_file)
        else:
            logger.info("loading configuration file %s from cache at %s", config_file, resolved_config_file)

        return config_dict, kwargs

    @classmethod
    def _dict_from_json_file(cls, json_file: str):
        """_dict_from_json_file"""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class PretrainedModel(nn.Cell):
    """
    Abstract class for Pretrained models
    """
    config_class = None
    pretrained_model_archive_map = {}
    base_model_prefix = ""

    def __init__(self, config):
        super().__init__(config)
        # Save config in model
        self.config = config

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).

        self.init_model_weights()
        self._backward_compatibility_gradient_checkpointing()
        """
        raise NotImplementedError

    def init_model_weights(self):
        """
        initialize model weights.
        """
        raise NotImplementedError

    @property
    def base_model(self):
        """
        to get base_model
        """
        return getattr(self, self.base_model_prefix, self)

    def get_input_embeddings(self) -> "nn.Cell":
        """
        Returns the model's input embeddings.

        Returns:
            :obj:`nn.Cell`: A mindspore cell mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        raise NotImplementedError

    def set_input_embeddings(self, value: "nn.Cell"):
        """
        Set model's input embeddings.

        Args:
            value (:obj:`nn.Cell`): A mindspore cell mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        resize the model position embeddings if necessary
        """
        raise NotImplementedError(
            f"`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__}"
        )

    def get_output_embeddings(self):
        """ Get model's output embeddings
            Return None if the model doesn't have output embeddings
        """
        return None  # Overwrite for models with output embeddings

    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """
        raise NotImplementedError(
            f"`get_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__}"
        )

    def tie_weights(self):
        """
        Make sure we are sharing the input and output embeddings.
        If you need this feature,
        you need to get it yourself output Add the output you need to add to the embeddings function_ Embedding layer,
        otherwise you cannot
        """
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None:
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """ Tie or clone module weights depending of weither we are using or not
        """

        output_embeddings.embedding_table = input_embeddings.embedding_table

        if hasattr(output_embeddings, "bias") and output_embeddings.bias is not None:
            output_embeddings.bias.data = ops.pad(
                output_embeddings.bias.data,
                (0, output_embeddings.embedding_table.shape[0] - output_embeddings.bias.shape[0]),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def resize_token_embeddings(self, new_num_tokens=None):
        """ Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        Take care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:

            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end.
                Reducing the size will remove vectors from the end.
                If not provided or None:
                    does nothing and just returns a pointer to
                    the input tokens ``mindspore.nn.Embeddings`` Module of the model.

        Return: ``mindspore.nn.Embeddings``
            Pointer to the input tokens Embeddings Module of the model
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed
        model_embeds = base_model.resize_tokenizer_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens

        # Tie weights again if needed
        # self.tie_weights()

        return model_embeds

    def resize_tokenizer_embeddings(self, new_num_tokens):
        """
        Obtain a new embedding layer or use the original one without updating it.
        """
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)
        return self.get_input_embeddings()

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
        """ Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``mindspore.nn.Embeddings``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.embedding_table.shape
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.embedding_table.data[:num_tokens_to_copy, :] = old_embeddings.embedding_table.data[
                                                                      :num_tokens_to_copy, :]

        return new_embeddings

    def save(self, save_dir: Union[str, os.PathLike]):
        "save pretrain model"
        raise NotImplementedError

    @classmethod
    def load(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
             *args, **kwargs):
        """
        Load a pre-trained checkpoint from a pre-trained model file or url,
        download and cache the pre-trained model file if model name in model list.

        Params:
            pretrained_model_name_or_path:
        """

        # Todo: load huggingface checkpoint
        config = kwargs.pop("config", None)
        # load config
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config = cls.config_class.load(config_path)
        model = cls(config, *args, **kwargs)
        if os.path.exists(pretrained_model_name_or_path):
            # File exists.
            model_file = os.path.join(pretrained_model_name_or_path)
            assert os.path.isfile(model_file)
        else:
            # Something unknown
            raise ValueError(
                f"unable to parse {pretrained_model_name_or_path} as a local path or model name")
        # load ckpt
        try:
            param_dict = load_checkpoint(model_file)
        except Exception as exc:
            raise ValueError(f"File {model_file} is not a checkpoint file, "
                             f"please check the path.") from exc

        param_not_load = load_param_into_net(model, param_dict)
        if len(param_not_load) == len(model.trainable_params()):
            raise KeyError(f"The following weights in model are not found: {param_not_load}")

        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """from_pretrained"""
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        kwargs.pop("from_tf", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
            elif os.path.isdir(pretrained_model_name_or_path):
                archive_file = os.path.join(pretrained_model_name_or_path, "mindspore_model.ckpt")
            elif os.path.isfile(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path

            # redirect to the cache, if necessary
            try:
                resolved_archive_file = str(cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    proxies=proxies,
                ))
            except EnvironmentError as exc:
                if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                    msg = f"Couldn't reach server at '{archive_file}' to download pretrained weights."
                else:
                    format1 = ", ".join(cls.pretrained_model_archive_map.keys())
                    format2 = ["mindspore_model.ckpt"]
                    msg = (
                        f"Model name '{pretrained_model_name_or_path}' "
                        f"was not found in model name list ({format1}). "
                        f"We assumed '{archive_file}' "
                        f"was a path or url to model weight files named one of {format2} but "
                        f"couldn't find any such file at this path or url."
                    )
                raise EnvironmentError(msg) from exc

            if resolved_archive_file == archive_file:
                logger.info("loading weights file %s", archive_file)
            else:
                logger.info("loading weights file %s from cache at %s", archive_file, resolved_archive_file)
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None:
            try:
                state_dict = mindspore.load_checkpoint(resolved_archive_file)
            except Exception as exc:
                raise OSError(
                    "Unable to load weights from mindspore checkpoint file. "
                ) from exc

        mindspore.load_param_into_net(model, state_dict)

        return model
