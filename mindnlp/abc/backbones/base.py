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
        """Get targets from either the sample or the net's output."""
        return sample["target"]

    def load_parameters(self, parameter_dict, strict_load, *args, **kwargs):
        """Copies parameters and buffers from parameter_dict into this module and its descendants."""
        raise NotImplementedError

    def get_context(self, src_tokens, mask=None):
        """Get Context from encoder."""
        raise NotImplementedError

    @classmethod
    def load_pretrained(cls, model_name_or_path, ckpt_file_name="model.ckpt", data_name_or_path="."):
        """Load model from a pretrained model file"""
        raise NotImplementedError

    def load_checkpoint(self, ckpt_file_name, net=None, strict_load=False, filter_prefix=None,
                        dec_key=None, dec_mode="AES-GCM", specify_prefix=None):
        """Load checkpoint from a specified file"""
        mindspore.load_checkpoint(ckpt_file_name, net=net, strict_load=strict_load,
                                  filter_prefix=filter_prefix, dec_key=dec_key,
                                  dec_mode=dec_mode, specify_prefix=specify_prefix)

    def save_checkpoint(self, save_obj, ckpt_file_name, integrated_save=True, async_save=False,
                        append_dict=None, enc_key=None, enc_mode="AES-GCM"):
        """Save checkpoint to a specified file"""
        mindspore.save_checkpoint(save_obj, ckpt_file_name, integrated_save=integrated_save,
                                  async_save=async_save, append_dict=append_dict,
                                  enc_key=enc_key, enc_mode=enc_mode)
