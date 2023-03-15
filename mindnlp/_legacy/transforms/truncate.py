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

# pylint: disable=import-outside-toplevel
# pylint: disable=c-extension-no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-boolean-expressions
"""
Transforms to process sequence
"""
import numpy as np
from mindspore.dataset.transforms.transforms import PyTensorOperation
from mindspore.dataset.text.transforms import Implementation


class Truncate(PyTensorOperation):
    """
    Truncate the input sequence.

    Args:
        max_seq_length (int): Maximum length required.

    Raises:
        TypeError: If `max_length` is not of type int.

    Examples:

    """

    def __init__(self, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.implementation = Implementation.PY

    def __call__(self, text_input):
        """
        Call method for input conversion for eager mode with C++ implementation.
        """
        if not isinstance(text_input, np.ndarray):
            raise TypeError(
                f"Input should be a text line in 1-D NumPy format, got {type(text_input)}.")
        return super().__call__(text_input)

    def execute_py(self, text_input):
        """
        Execute method.
        """
        return self._execute_py(text_input)

    def _execute_py(self, text_input):
        """
        Execute method.
        """
        return text_input[:self.max_seq_length]
