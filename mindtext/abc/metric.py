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
Abstract class for Metrics
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from mindspore import Tensor

class Metric(metaclass=ABCMeta):
    """
    Base class of all metrics. Never use this class directly, but instantiate one of its subclasses instead.

    Functions `update` will accumulate intermediate results in the evaluation process, `eval` will evaluate the final
    result, and `clear` will reinitialize the intermediate results. Function `get_metric_name` will provide class name.

    """
    def __init__(self):
        pass

    def _convert_data_type(self, data):
        """
        Convert data type to numpy array.

        Args:
            data (Object): Input data.

        Returns:
            - **data** (np.ndarray) - Data with `np.ndarray` type.

        Raises:
            TypeError: If `data` is not a tensor, list or numpy.ndarray.

        """
        if isinstance(data, Tensor):
            data = data.asnumpy()
        elif isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            pass
        else:
            raise TypeError(f'For class `Metrics` and its derived classes, the input data type must be tensor, list or'
                            f' numpy.ndarray, but got {type(data)}.')
        return data

    @abstractmethod
    def clear(self):
        """
        An interface describes the behavior of clearing the internal evaluation result. All subclasses of `Metrics`
        must override this interface.

        Raises:
            NotImplementedError: If this interface is called.

        """
        raise NotImplementedError(f'Function `clear` not implemented in {self.__class__.__name__}')

    @abstractmethod
    def eval(self):
        """
        An interface describes the behavior of computing the evaluation result. All subclasses of `Metrics`
        must override this interface.

        Raises:
            NotImplementedError: If this interface is called.

        """
        raise NotImplementedError(f'Function `eval` not implemented in {self.__class__.__name__}')

    @abstractmethod
    def updates(self, preds, labels):
        """
        An interface describes the behavior of updating the internal evaluation result. All subclasses of `Metrics`
        must override this interface.

        Raises:
            NotImplementedError: If this interface is called.

        """
        raise NotImplementedError(f'Function `updates` not implemented in {self.__class__.__name__}')

    @abstractmethod
    def get_metric_name(self):
        """
        An interface returns the name of the metric. All subclasses of `Metrics` must override this interface.

        Raises:
            NotImplementedError: If this interface is called.

        """
        raise NotImplementedError(f'Function `get_metric_name` not implemented in {self.__class__.__name__}')
