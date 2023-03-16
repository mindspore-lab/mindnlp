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
"""util function"""

from collections.abc import Iterable
import numpy as np
from mindspore import Tensor

def _check_value_type(arg_name, arg_value, valid_types):
    """
    Checks whether the data type of argument is valid

    Args:
        arg_name (str): Name of the argument validated.
        arg_value (Object): Value of the argument validated.
        valid_types (list): Valid data types of the argument.

    Returns:
        - **arg_value** (Object) - Value of the argument validated.

    Raises:
        TypeError: If the data type of the argument is not valid.

    """
    valid_types = valid_types if isinstance(valid_types, Iterable) else (valid_types,)
    num_types = len(valid_types)
    if isinstance(arg_value, bool) and bool not in tuple(valid_types):
        raise TypeError(f'Type of `{arg_name}` should be {"one of " if num_types > 1 else ""}'
                        f' `{valid_types if num_types > 1 else str(valid_types[0])}`, '
                        f'but got `{arg_value}` with type `{type(arg_value).__name__}`.')
    if not isinstance(arg_value, tuple(valid_types)):
        raise TypeError(f'Type of `{arg_name}` should be {"one of " if num_types > 1 else ""}'
                        f'`{valid_types if num_types > 1 else str(valid_types[0])}`, '
                        f'but got `{arg_value}` with type `{type(arg_value).__name__}`.')
    return arg_value

def _check_onehot_data(data):
    """
    Checks whether input data is one-hot encoding.

    Args:
        data (np.array): Input data.

    Returns:
        - **ans** (bool) - Rreturn true, if input data is one-hot encoding.
    """
    ans = False
    data = _convert_data_type(data)
    if np.equal(data ** 2, data).all():
        shp = (data.shape[0],) + data.shape[2:]
        if np.equal(np.ones(shp), data.sum(axis=1)).all():
            ans = True
            return ans
    return ans

def _convert_data_type(data):
    """
    Converts data type to numpy array.

    Args:
        data (Union[Tensor, list, np.ndarray]): Input data.

    Returns:
        - **data** (np.ndarray) - Data with `np.ndarray` type.

    Raises:
        TypeError: If the data type is not tensor, list or np.ndarray.

    """
    if isinstance(data, Tensor):
        data = data.asnumpy()
    elif isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise TypeError(f'Input data type must be tensor, list or '
                        f'np.ndarray, but got {type(data)}.')
    return data

def _check_shape(y_pred, y_true, n_class=None):
    """
    Checks the shapes of y_pred and y_true.

    Args:
        y_pred (Tensor): Predict tensor.
        y_true (Tensor): Target tensor.
    """
    if y_pred.ndim != y_true.ndim + 1 and n_class != 1:
        raise ValueError(f'The dimension of `y_pred` should be equal to the dimension of `y_true` '
                         f'add 1, but got `y_pred` dimension: {y_pred.ndim} and `y_true` dimension:'
                         f' {y_true.ndim}.')
    if y_true.shape != (y_pred.shape[0],) + y_pred.shape[2:] and n_class != 1:
        raise ValueError(f'`y_pred` shape and `y_true` shape can not match, `y_true` shape should '
                         f'be equal to `y_pred` shape that the value at index 1 is deleted. Such as'
                         f' `y_pred` shape (1, 2, 3), then `y_true` shape should be (1, 3). But got'
                         f' `y_pred` shape {y_pred.shape} and `y_true` shape {y_true.shape}.')


def _get_rank(raw_list):
    raw_x = np.array(raw_list)
    rank_x = np.empty(raw_x.shape, dtype=int)
    sort_x = np.argsort(-raw_x)
    for i, k in enumerate(sort_x):
        rank_x[k] = i + 1
    return rank_x
