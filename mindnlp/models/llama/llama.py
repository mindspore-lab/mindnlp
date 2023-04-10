# Copyright (c) Meta Platforms, Inc. and affiliates.
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

# This software may be used and distributed according to the terms of the GNU General Public License version 3.
""" MindNLP llama model."""

from mindspore import nn, ops, Parameter

class RMSNorm(nn.Cell):
    '''
    RMSNorm
    '''
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(ops.ones(dim))

    def _norm(self, norm_x):
        return norm_x * ops.rsqrt(ops.mean(ops.pow(norm_x, 2), -1, keep_dims=True) + self.eps)

    def construct(self, construct_x):
        '''RMSNorm construct'''
        output = self._norm(construct_x.float()).astype(construct_x.dtype)
        return output * self.weight
