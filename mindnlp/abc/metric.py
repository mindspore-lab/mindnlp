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

class Metric(metaclass=ABCMeta):
    """
    Base class of all metrics. Never use this class directly, but instantiate one of
    its subclasses instead.

    Functions `update` will accumulate intermediate results in the evaluation process,
    `eval` will evaluate the final result, and `clear` will reinitialize the intermediate
    results. Function `get_metric_name` will provide class name.

    """
    def __init__(self):
        pass

    @abstractmethod
    def clear(self):
        """
        An interface describes the behavior of clearing the internal evaluation result.
        All subclasses of `Metrics` must override this interface.

        Raises:
            NotImplementedError: If this interface is called.

        """
        raise NotImplementedError(f'Function `clear` not implemented in {self.__class__.__name__}')

    @abstractmethod
    def eval(self):
        """
        An interface describes the behavior of computing the evaluation result.
        All subclasses of `Metrics` must override this interface.

        Raises:
            NotImplementedError: If this interface is called.

        """
        raise NotImplementedError(f'Function `eval` not implemented in {self.__class__.__name__}')

    @abstractmethod
    def update(self, *inputs):
        """
        An interface describes the behavior of updating the internal evaluation result.
        All subclasses of `Metrics` must override this interface.

        Args:
            inputs: Variable parameter list.

        Raises:
            NotImplementedError: If this interface is called.

        """
        raise NotImplementedError(f'Function `update` not implemented in {self.__class__.__name__}')

    @abstractmethod
    def get_metric_name(self):
        """
        An interface returns the name of the metric. All subclasses of `Metrics` must
        override this interface.

        Raises:
            NotImplementedError: If this interface is called.

        """
        raise NotImplementedError(f'Function `get_metric_name` not implemented '
                                  f'in {self.__class__.__name__}')
