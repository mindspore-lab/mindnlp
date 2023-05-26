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
Pretrained config.
"""

import copy
import json
import os
from typing import Optional, Tuple, Dict
from mindspore import log as logger

from mindnlp.configs import HF_CONFIG_URL_BASE
from mindnlp.utils.download import cached_path

class PreTrainedConfig:
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
        return cls.from_pretrained(pretrained_model_name_or_path)

    @property
    def use_return_dict(self) -> bool:
        """
        `bool`: Whether or not return [`~utils.ModelOutput`] instead of tuples.
        """
        # If torchscript is set, force `return_dict=False` to avoid jit errors
        return self.return_dict

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "PreTrainedConfig":
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
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> "PreTrainedConfig":
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
        _ = kwargs.pop("force_download", False)
        _ = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        _ = kwargs.pop("local_files_only", False)
        from_pt = kwargs.pop("from_pt", False)

        folder_name = None
        if pretrained_config_archive_map is None:
            pretrained_config_archive_map = cls.pretrained_config_archive_map

        if pretrained_model_name_or_path in pretrained_config_archive_map:
            config_file = pretrained_config_archive_map[pretrained_model_name_or_path]
            folder_name = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        elif os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        elif from_pt:
            config_file = HF_CONFIG_URL_BASE.format(pretrained_model_name_or_path)
            folder_name = pretrained_model_name_or_path
        else:
            raise ValueError(f'not found config of {pretrained_model_name_or_path}')

        try:
            # Load from URL or cache if already cached
            resolved_config_file = str(cached_path(
                config_file,
                cache_dir=cache_dir,
                proxies=proxies,
                folder_name=folder_name
            )[0])

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
                    f"is a correct model identifier listed on 'https://download.mindspore.cn/toolkits/mindnlp/models'\n\n"
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
