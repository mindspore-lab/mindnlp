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
'''
    cuda_specs
'''
import dataclasses
from typing import Optional, Tuple
import subprocess
import re
from mindspore import context


@dataclasses.dataclass(frozen=True)
class CUDASpecs:
    cuda_version_string: str
    cuda_version_tuple: Tuple[int, int]


def get_cuda_version_tuple() -> Tuple[int, int]:
    result = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, text=True, check=True)
    match = re.search(r"V(\d+)\.(\d+)", result.stdout)
    if match:
        major, minor = map(int, match.groups())
        return major, minor
    return 0, 0


def get_cuda_version_string() -> str:
    major, minor = get_cuda_version_tuple()
    return f"{major}{minor}"


def get_cuda_specs() -> Optional[CUDASpecs]:
    if not context.get_context("device_target") == "GPU":
        return None

    return CUDASpecs(
        cuda_version_string=(get_cuda_version_string()),
        cuda_version_tuple=get_cuda_version_tuple(),
    )
