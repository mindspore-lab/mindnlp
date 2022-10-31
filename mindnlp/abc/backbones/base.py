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
"""Basic class for models"""

import mindspore
from mindspore import nn


class BaseModel(nn.Cell):
    r"""Basic class for models"""

    @classmethod
    def add_attrs(cls, parser):
        """Add model-specific arguments to the parser."""
        raise NotImplementedError

    def get_targets(self, sample):
        """
        Get targets from either the sample or the net's output.

        Args:
            sample (dict): The sample or the net's output.

        Returns:
            Tensor, the net's target.
        """
        return sample["target"]

    def load_parameters(self, parameter_dict, strict_load, *args, **kwargs):
        """
        Copies parameters and buffers from parameter_dict into this module and its descendants.

        Args:
            parameter_dict (dict): A dictionary consisting of key: parameters's name, value: parameter.
            strict_load (bool): Whether to strict load the parameter into net. If False, it will load
                parameter into net when parameter name's suffix in checkpoint file is the same as the
                parameter in the network. When the types are inconsistent perform type conversion on
                the parameters of the same type, such as float32 to float16.
        """
        raise NotImplementedError

    def get_context(self, src_tokens, mask=None):
        """
        Get Context from encoder.

        Args:
            src_tokens (Tensor): Tokens of source sentences.
            mask (Tensor): Its elements identify whether the corresponding input token is padding or not.
                If True, not padding token. If False, padding token. Defaults to None.
        """
        raise NotImplementedError

    @classmethod
    def load_pretrained(cls, model_name_or_path, ckpt_file_name="model.ckpt", data_name_or_path="."):
        """Load model from a pretrained model file"""
        raise NotImplementedError

    def load_checkpoint(self, ckpt_file_name, net=None, strict_load=False, filter_prefix=None,
                        dec_key=None, dec_mode="AES-GCM", specify_prefix=None):
        """
        Load checkpoint from a specified file.

        Args:
            ckpt_file_name (str): Checkpoint file name.
            net (Cell): The network where the parameters will be loaded. Default: None
            strict_load (bool): Whether to strict load the parameter into net. If False, it will
                load parameter into net when parameter name's suffix in checkpoint file is the same
                as the parameter in the network. When the types are inconsistent perform type conversion
                on the parameters of the same type, such as float32 to float16. Default: False.
            filter_prefix (Union[str, list[str], tuple[str]]): Parameters starting with the filter_prefix
                will not be loaded. Default: None.
            dec_key (Union[None, bytes]): Byte type key used for decryption. If the value is None, the
                decryption is not required. Default: None.
            dec_mode (str): This parameter is valid only when dec_key is not set to None. Specifies the
                decryption mode, currently supports 'AES-GCM' and 'AES-CBC'. Default: 'AES-GCM'.
            specify_prefix (Union[str, list[str], tuple[str]]): Parameters starting with the specify_prefix
                will be loaded. Default: None.
        """
        mindspore.load_checkpoint(ckpt_file_name, net=net, strict_load=strict_load,
                                  filter_prefix=filter_prefix, dec_key=dec_key,
                                  dec_mode=dec_mode, specify_prefix=specify_prefix)

    def save_checkpoint(self, save_obj, ckpt_file_name, integrated_save=True, async_save=False,
                        append_dict=None, enc_key=None, enc_mode="AES-GCM"):
        """
        Save checkpoint to a specified file.

        Args:
            save_obj (Union[Cell, list]): The cell object or data list(each element is a dictionary,
                like [{“name”: param_name, “data”: param_data},…], the type of param_name would be string,
                and the type of param_data would be parameter or Tensor).
            ckpt_file_name (str): Checkpoint file name. If the file name already exists, it will be overwritten.
            integrated_save (bool): Whether to integrated save in automatic model parallel scene. Default: True
            async_save (bool): Whether to open an independent thread to save the checkpoint file. Default: False
            append_dict (dict): Additional information that needs to be saved. The key of dict must be str, the
                value of dict must be one of int, float, bool, string, Parameter or Tensor. Default: None.
            enc_key (Union[None, bytes]): Byte type key used for encryption. If the value is None, the encryption
                is not required. Default: None.
            enc_mode (str): This parameter is valid only when enc_key is not set to None. Specifies the encryption
                mode, currently supports 'AES-GCM' and 'AES-CBC'. Default: 'AES-GCM'.
        """
        mindspore.save_checkpoint(save_obj, ckpt_file_name, integrated_save=integrated_save,
                                  async_save=async_save, append_dict=append_dict,
                                  enc_key=enc_key, enc_mode=enc_mode)
