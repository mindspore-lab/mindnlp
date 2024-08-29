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
"""core module"""
import contextlib
import mindspore
from mindspore.common.api import _pynative_executor

class no_grad(contextlib.ContextDecorator):
    """
    Context Manager to disable gradient calculation. When enter this context, we will disable calculate
    gradient. When exit this context, we will resume its prev state.
    Currently, it can only use in Pynative mode. It also can be used as decorator.
    """

    def __enter__(self):
        if mindspore.get_context("mode") == mindspore.GRAPH_MODE:
            raise RuntimeError("For no_grad feature, currently only support Pynative mode, but got Graph mode.")
        _pynative_executor.set_enable_grad(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pynative_executor.set_enable_grad(True)
        return False

class enable_grad(contextlib.ContextDecorator):
    """
    Context Manager to disable gradient calculation. When enter this context, we will disable calculate
    gradient. When exit this context, we will resume its prev state.
    Currently, it can only use in Pynative mode. It also can be used as decorator.
    """

    def __enter__(self):
        if mindspore.get_context("mode") == mindspore.GRAPH_MODE:
            raise RuntimeError("For no_grad feature, currently only support Pynative mode, but got Graph mode.")
        _pynative_executor.set_enable_grad(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pynative_executor.set_enable_grad(False)
        return False
