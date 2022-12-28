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
import json
import os
from typing import Union, Optional
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import nn

class PretrainedConfig:
    """
    Abstract class for Pretrained models config.
    """
    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)

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
    def load(cls, pretrained_model_name_or_path):
        """load config."""
        if os.path.exists(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            raise ValueError(
                f"unable to parse {pretrained_model_name_or_path} as a local path or model name")

        config = cls.from_json(config_file)

        return config

class PretrainedModel(nn.Cell):
    """
    Abstract class for Pretrained models
    """
    config_class = None
    def __init__(self, config):
        super().__init__()
        self.config = config

    def init_model_weights(self):
        """
        initialize model weights.
        """
        raise NotImplementedError

    def get_input_embeddings(self) -> "nn.Cell":
        """
        Returns the model's input embeddings.

        Returns:
            :obj:`nn.Cell`: A mindspore cell mapping vocabulary to hidden states.
        """
        raise NotImplementedError

    def set_input_embeddings(self, value: "nn.Cell"):
        """
        Set model's input embeddings.

        Args:
            value (:obj:`nn.Cell`): A mindspore cell mapping vocabulary to hidden states.
        """
        raise NotImplementedError

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        resize the model position embeddings if necessary
        """
        raise NotImplementedError(
            f"`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__}"
        )

    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """
        raise NotImplementedError(
            f"`get_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__}"
        )

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
