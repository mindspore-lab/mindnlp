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
"""utils for mindspore backward compatibility."""
import mindspore
from packaging import version

MIN_COMPATIBLE_VERSION = '1.8.1'
MAX_GRAPH_FIRST_VERSION = '1.12.0'
API_COMPATIBLE_VERSION = '1.10.1'

MS_VERSION = mindspore.__version__
MS_VERSION = MS_VERSION.replace('rc', '')

less_min_minddata_compatible = version.parse(MS_VERSION) <= version.parse(MIN_COMPATIBLE_VERSION)
less_min_compatible = version.parse(MS_VERSION) < version.parse(MIN_COMPATIBLE_VERSION)
less_min_pynative_first = version.parse(MS_VERSION) <= version.parse(MAX_GRAPH_FIRST_VERSION)
less_min_api_compatible = version.parse(MS_VERSION) <= version.parse(API_COMPATIBLE_VERSION)

__all__ = [
    'less_min_compatible',
    'less_min_pynative_first',
    'less_min_api_compatible',
    'less_min_minddata_compatible'
]
