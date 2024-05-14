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
"""
evaluate module.
"""
from typing import Optional, Union
from datasets import DownloadConfig, DownloadMode
from datasets.utils.version import Version
from evaluate import config
from evaluate import load as eval_load
from evaluate.module import EvaluationModule

config.HUB_EVALUATE_URL = "https://openi.pcl.ac.cn/{path}/raw/branch/{revision}/{name}"

def load(
    path: str,
    config_name: Optional[str] = None,
    module_type: Optional[str] = None,
    process_id: int = 0,
    num_process: int = 1,
    cache_dir: Optional[str] = None,
    experiment_id: Optional[str] = None,
    keep_in_memory: bool = False,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[DownloadMode] = None,
    revision: Optional[Union[str, Version]] = None,
    **init_kwargs,
) -> EvaluationModule:
    return eval_load(
        path,
        config_name,
        module_type,
        process_id,
        num_process,
        cache_dir,
        experiment_id,
        keep_in_memory,
        download_config,
        download_mode,
        revision,
        **init_kwargs,
    )
