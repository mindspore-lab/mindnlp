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
"""AddToken transform"""
import numpy as np
from mindspore.dataset.transforms.transforms import PyTensorOperation
from mindspore.dataset.text.transforms import Implementation


class PadTransform(PyTensorOperation):
    """
    Pad tensor to a fixed length with given padding value.

    Args:
        max_length (int): Maximum length to pad to.
        pad_value (int): Value to pad the tensor with.
        return_length (bool): Whether return auxiliary sequence length.

    Raises:
        TypeError: If `token` is not of type str.

    Supported Platforms:
        ``CPU``

    Examples:

    """

    # @check_decode
    def __init__(self, max_length: int, pad_value:int, return_length:bool = False):
        super().__init__()
        self.max_length = max_length
        self.pad_value = pad_value
        self.return_length = return_length
        self.implementation = Implementation.PY

    def __call__(self, text_input):
        """
        Call method for input conversion for eager mode with C++ implementation.
        """
        if not isinstance(text_input, np.ndarray):
            raise TypeError(
                f"Input should be a text line in 1-D ndarray contains string, got {type(text_input)}.")
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
        text_input = text_input[:self.max_length]
        text_length = len(text_input)

        pad_value = np.array([self.pad_value] * (self.max_length - text_length), text_input.dtype)
        text_output = np.concatenate([text_input, pad_value], 0)

        if self.return_length:
            length = np.array(text_length)
            return text_output, length

        return text_output
