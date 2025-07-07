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
from mindspore.common.api import _pynative_executor
from ..utils._contextlib import _NoParamDecoratorContextManager, _DecoratorContextManager

class no_grad(_NoParamDecoratorContextManager):
    """
    Context Manager to disable gradient calculation. When enter this context, we will disable calculate
    gradient. When exit this context, we will resume its prev state.
    Currently, it can only use in Pynative mode. It also can be used as decorator.
    """
    def __init__(self) -> None:
        super().__init__()
        self.prev_state = False

    def __enter__(self):
        self.prev_state = _pynative_executor.enable_grad()
        _pynative_executor.set_enable_grad(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pynative_executor.set_enable_grad(self.prev_state)


class enable_grad(_NoParamDecoratorContextManager):
    """
    Context Manager to disable gradient calculation. When enter this context, we will disable calculate
    gradient. When exit this context, we will resume its prev state.
    Currently, it can only use in Pynative mode. It also can be used as decorator.
    """

    def __enter__(self):
        self.prev_state = _pynative_executor.enable_grad()
        _pynative_executor.set_enable_grad(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pynative_executor.set_enable_grad(self.prev_state)

class inference_mode(_DecoratorContextManager):
    """
    Context Manager to enable or disable inference mode.
    Currently, when enter this context, it is equivalent to enable_grad or no_grad.
    When exit this context, we will resume its prev state.
    Currently, it can only use in Pynative mode. It also can be used as decorator.
    """

    def __init__(self, mode: bool = True) -> None:
        super().__init__()
        self.mode = mode
        self.prev_state = False

    def __new__(cls, mode: bool = True):
        if isinstance(mode, bool):
            return super().__new__(cls)
        return cls()(mode)

    def __enter__(self) -> None:
        self.prev_state = _pynative_executor.enable_grad()
        _pynative_executor.set_enable_grad(not self.mode)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _pynative_executor.set_enable_grad(self.prev_state)

    def clone(self) -> "inference_mode":
        return self.__class__(self.mode)