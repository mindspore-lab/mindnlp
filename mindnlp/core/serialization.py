"""serialization"""
from typing import OrderedDict
import numpy as np
from mindspore import Tensor
from mindspore.train.serialization import _exec_save, _parse_ckpt_proto, tensor_to_np_type, tensor_to_ms_type
from .nn import Module

def get_data_list(param_dict):
    """Get state dict of the Peft model for saving."""
    data_list = OrderedDict()  # {key: [dims, tensor_type, data]}

    for key, value in param_dict.items():
        data_list[key] = []
        dims = []
        if value.shape == ():
            dims.append(0)
        else:
            for dim in value.shape:
                dims.append(dim)
        data_list[key].append(dims)
        tensor_type = str(value.dtype)
        data_list[key].append(tensor_type)
        data = value.asnumpy().reshape(-1)
        data_list[key].append(data)

    return data_list

def save_checkpoint(save_obj, ckpt_file_name):
    r"""
    Save checkpoint to a specified file.
    """
    if isinstance(save_obj, Module):
        data_list = get_data_list(save_obj.parameters_dict())
    elif isinstance(save_obj, dict):
        data_list = get_data_list(save_obj)
    else:
        raise ValueError(f'not support save object {type(save_obj)}')
    _exec_save(ckpt_file_name, data_list)

def load_checkpoint(ckpt_file_name):
    """
    Load checkpoint info from a specified file.
    """
    try:
        checkpoint_list = _parse_ckpt_proto(ckpt_file_name, None, None) # pylint: disable=no-value-for-parameter
    except:
        checkpoint_list = _parse_ckpt_proto(ckpt_file_name, None, None, None)

    parameter_dict = {}
    try:
        param_data_list = []

        for element_id, element in enumerate(checkpoint_list.value):
            if element.tag == "random_op":
                parameter_dict["random_op"] = element.tensor.tensor_content
                continue

            data = element.tensor.tensor_content
            data_type = element.tensor.tensor_type
            np_type = tensor_to_np_type.get(data_type)
            ms_type = tensor_to_ms_type[data_type]
            if data_type == 'str':
                str_length = int(len(data) / 4)
                np_type = np_type + str(str_length)
            if data_type == "BFloat16":
                dims = element.tensor.dims
                param_data = np.frombuffer(data, np_type)
                param_data = param_data.reshape(list(dims))
                parameter = Tensor(param_data, ms_type)
                parameter_dict[element.tag] = parameter
                continue
            element_data = np.frombuffer(data, np_type)
            param_data_list.append(element_data)
            if (element_id == len(checkpoint_list.value) - 1) or \
                    (element.tag != checkpoint_list.value[element_id + 1].tag):
                new_data = b"".join(param_data_list)
                param_data = np.frombuffer(new_data, np_type)
                param_data_list.clear()
                dims = element.tensor.dims
                if dims == [0] and data_type == 'str':
                    parameter_dict[element.tag] = str(element_data[0])
                else:
                    if dims == [0] and 'Float' in data_type:
                        param_data = float(param_data[0])
                    if dims == [0] and 'Int' in data_type:
                        param_data = int(param_data[0])
                    if dims not in ([0], [1]):
                        param_data = param_data.reshape(list(dims))
                    parameter = Tensor(param_data, ms_type)
                    parameter_dict[element.tag] = parameter

    except BaseException as e:
        raise ValueError(e.__str__() + "\nFor 'load_checkpoint', "
                                       "failed to load the checkpoint file {}.".format(ckpt_file_name)) from e

    if not parameter_dict:
        raise ValueError("The loaded parameter dict is empty after filter or specify, please check whether "
                         "'filter_prefix' or 'specify_prefix' are set correctly.")

    return parameter_dict
