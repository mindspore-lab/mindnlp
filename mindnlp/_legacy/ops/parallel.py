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
"""Parallel Operators"""
from mindspore import Parameter
from mindspore.ops import AllGather as msAllGather
from mindspore.ops.primitive import _run_op

class AllGather(msAllGather):
    """AllGather op."""
    def __call__(self, *args):
        should_elim, output = self.check_elim(*args)
        for arg in args:
            if isinstance(arg, Parameter) and arg.has_init:
                arg.init_data()
        if should_elim:
            return output
        return _run_op(self, self.name, args)

__all__ = ['AllGather']
