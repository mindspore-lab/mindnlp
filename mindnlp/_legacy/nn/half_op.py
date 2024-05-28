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
"""nn wraper operators for half precision cast."""

from mindspore import nn, ops

class Matmul(nn.Cell):
    r"""
    Matmul Operation
    """
    def construct(self, a, b):
        r"""
        This method constructs a new matrix by performing matrix multiplication between the input matrices 'a' and 'b'.
        
        Args:
            self (object): The instance of the Matmul class.
            a (matrix): The first input matrix for the matrix multiplication operation.
            b (matrix): The second input matrix for the matrix multiplication operation.
        
        Returns:
            None: This method does not return any value directly, but the result of the matrix multiplication operation can be obtained using the 'ops.matmul' method.
        
        Raises:
            - TypeError: If the input 'a' or 'b' is not a valid matrix object.
            - ValueError: If the dimensions of the input matrices are not compatible for matrix multiplication.
        """
        return ops.matmul(a, b)

__all__ = ['Matmul']
