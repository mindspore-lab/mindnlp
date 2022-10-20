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
"""Custom functional api for legacy mindspore"""
import mindspore
from packaging import version
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim

def _kl_div(inputs, target, reduction='none', log_target=False):
    """KLDiv function."""
    if log_target:
        kl_div_loss = ops.exp(target) * (target - inputs)
    else:
        output = target * (ops.log(target) - inputs)
        zeros = ops.zeros_like(inputs)
        kl_div_loss = ops.select(target > 0, output, zeros)
    if reduction == 'sum':
        return kl_div_loss.sum()
    if reduction == 'mean':
        return kl_div_loss.mean()
    return kl_div_loss

def _softmax(inputs, axis=-1):
    _softmax_op = _get_cache_prim(ops.Softmax)(axis)
    return _softmax_op(inputs)

if version.parse(mindspore.__version__) <= version.parse('1.8.1'):
    softmax = _softmax
    kl_div = _kl_div
else:
    softmax = ops.softmax
    kl_div = ops.kl_div
