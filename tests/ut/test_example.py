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
Test case example
"""
from packaging import version

def test_import_mindspore():
    """
    Feature: What feature you test
    Description: What input in what scene
    Expectation: success or throw xxx exception or result == xxx, etc.
    """
    import mindspore
    assert version.parse(mindspore.__version__) >= version.parse('1.7.0')
