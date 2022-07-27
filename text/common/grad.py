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
"""Grad"""
from mindspore.ops import stop_gradient, GradOperation

grad_func = GradOperation(True, False, False)
grad_cell = GradOperation(False, True, False)
def value_and_grad(fn, params=None, has_aux=False):
    """value_and_grad"""
    if params is None:
        grad_ = grad_func
    else:
        grad_ = grad_cell

    def fn_aux(*args):
        outputs = fn(*args)
        no_grad_outputs = ()
        for out in outputs[1:]:
            no_grad_outputs += (stop_gradient(out),)
        return outputs[0], no_grad_outputs

    if has_aux:
        fn_ = fn_aux
    else:
        fn_ = fn

    def value_and_grad_f(*args):
        values = fn_(*args)
        if params is None:
            grads = grad_(fn_)(*args)
        else:
            grads = grad_(fn_, params)(*args)
        return values, grads
    return value_and_grad_f

def grad(fn, params=None, has_aux=False):
    """grad"""
    value_and_grad_f = value_and_grad(fn, params, has_aux)
    def grad_f(*args):
        _, g = value_and_grad_f(*args)
        return g
    return grad_f
