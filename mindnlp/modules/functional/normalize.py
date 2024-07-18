# Copyright 2024 Huawei Technologies Co., Ltd
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
MindNLP defined functional methods
"""
from mindspore import ops

def normalize(input, p=2.0, dim=1):
    r"""
    Normalize a tensor along a specified dimension.
    
    Args:
        input (Tensor): The input tensor to be normalized.
        p (float, optional): The power parameter for the normalization. Default is 2.0.
        dim (int, optional): The dimension along which to normalize the tensor. Default is 1.
    
    Returns:
        None
    
    Raises:
        TypeError: If the input is not a tensor.
        ValueError: If the specified dimension is out of range or if the power parameter is not a positive number.
    
    This function normalizes the input tensor along the specified dimension using the power parameter 'p'.
    The normalization is performed by dividing each element of the tensor by the Lp norm of the tensor along the specified dimension.
    The Lp norm is defined as the p-th root of the sum of the absolute values raised to the power of 'p'.
    The resulting tensor will have the same shape as the input tensor.
    """
    return input / ops.norm(input, ord=p, dim=dim, keepdim=True)
