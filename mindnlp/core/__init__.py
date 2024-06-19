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
import functools
import mindspore
from .tensor import Tensor


def core_patch_decorator(autocast_instance, func):
    @functools.wraps(func)
    def decorate_autocast(*args, **kwargs):
        with autocast_instance:
            return func(*args, **kwargs)

    return decorate_autocast

class CorePatch:
    def __enter__(self):
        # Replace StubTensor with Tensor
        self.original_stub_tensor = mindspore.common._stub_tensor.StubTensor
        mindspore.common._stub_tensor.StubTensor = Tensor
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original StubTensor
        mindspore.common._stub_tensor.StubTensor = self.original_stub_tensor

    def __call__(self, func):
        return core_patch_decorator(self, func)

__all__ = ['Tensor', 'CorePatch']
