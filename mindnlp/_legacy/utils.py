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
# pylint: disable=W0702
""""legacy utils"""
import numpy as np

from mindspore import Tensor
from mindspore.train.serialization import _parse_ckpt_proto, \
    tensor_to_np_type, tensor_to_ms_type

try:
    from mindspore.train.serialization import _load_mapparameter as _load_map_parameter
except:
    from mindspore.train.serialization import _load_map_parameter

from mindnlp.utils import logging

logger = logging.get_logger(__name__)


def load_checkpoint(ckpt_file_name):
    """redefined load_checkpoint method, not mindspore official version."""
    logger.info("Execute the process of loading checkpoint files.")
    checkpoint_list = _parse_ckpt_proto(ckpt_file_name, None, None)
    parameter_dict = {}
    # try:
    param_data_list = []
    for element_id, element in enumerate(checkpoint_list.value):
        if element.tag == "random_op":
            parameter_dict["random_op"] = element.tensor.tensor_content
            continue
        if element.tensor.ByteSize() == 0:
            _load_map_parameter(element, parameter_dict)
            continue
        data = element.tensor.tensor_content
        data_type = element.tensor.tensor_type
        np_type = tensor_to_np_type.get(data_type)
        ms_type = tensor_to_ms_type[data_type]
        if data_type == 'str':
            str_length = int(len(data) / 4)
            np_type = np_type + str(str_length)
        # element_data = np.frombuffer(data, np_type)
        param_data_list.append(data)
        if (element_id == len(checkpoint_list.value) - 1) or \
                (element.tag != checkpoint_list.value[element_id + 1].tag):
            # param_data = np.concatenate((param_data_list), axis=0)
            new_data = b''.join(param_data_list)

            param_data = np.frombuffer(new_data, np_type)
            param_data_list.clear()
            dims = element.tensor.dims

            if dims == [0] and 'Float' in data_type:
                param_data = float(param_data[0])
            if dims == [0] and 'Int' in data_type:
                param_data = int(param_data[0])
            if dims not in ([0], [1]):
                param_data = np.lib.stride_tricks.as_strided(param_data, list(dims))
            parameter = Tensor(param_data, ms_type)
            parameter_dict[element.tag] = parameter

        logger.info("Loading checkpoint files process is finished.")

    if not parameter_dict:
        raise ValueError("The loaded parameter dict is empty after filter or specify, please check whether "
                         "'filter_prefix' or 'specify_prefix' are set correctly.")

    return parameter_dict
