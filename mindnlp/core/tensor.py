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
"""mindnlp tensor"""
from mindspore import Tensor as MSTensor
from mindspore._c_expression import TensorNode # pylint: disable=no-name-in-module

class Tensor:
    tensor = None
    stub = None
    def __init__(self, input, dtype=None):
        if isinstance(input, TensorNode):
            self.stub = input
        elif isinstance(input, Tensor):
            self.tensor = input
        else:
            self.tensor = MSTensor(input, dtype=dtype)

    @property
    def data(self):
        return self.tensor

    def stub_sync(self):
        """sync real tensor."""
        if self.stub:
            val = self.stub.get_value()
            self.tensor = MSTensor(val, internal=True)
            if hasattr(self, "member_cache"):
                for k, v in self.member_cache.items():
                    setattr(self.tensor, k, v)
            self.stub = None
        return self.tensor
