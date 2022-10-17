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
"""
Abstract class for Register
"""

from functools import wraps

class Register():
    """Register abstract class"""
    def __init__(self, name, map_rule):
        self.name = name
        self.mem_dict = {}
        self.map_rule = map_rule

    def register(self, func):
        """register function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            dataset = func(*args, **kwargs)
            return dataset
        name = self.map_rule(func)
        self.mem_dict[name] = wrapper
        return wrapper

    def __call__(self, name, *args, **kwargs):
        lname = name.lower()
        if lname not in self.mem_dict:
            raise ValueError(f'{name} is not registered. Please check the dataset list.')
        return self.mem_dict[name.lower()](*args, **kwargs)
