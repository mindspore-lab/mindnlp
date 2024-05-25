# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""multitask prompt tuning model"""
import mindspore
from mindspore import ops, Parameter

from ..prompt_tuning import PromptEmbedding
from ...utils import TaskType

from .config import MultitaskPromptTuningConfig, MultitaskPromptTuningInit


# This code is adapted for the paper: https://arxiv.org/abs/2303.02861 and
# constitutes the work done at MIT-IBM Watson Research Lab.


class MultitaskPromptEmbedding(PromptEmbedding):
    def __init__(self, config: MultitaskPromptTuningConfig, word_embeddings):
        super().__init__(config, word_embeddings)

        self.num_tasks = config.num_tasks
        self.num_ranks = config.num_ranks
        self.num_virtual_tokens = config.num_virtual_tokens

        self.num_transformer_submodules = config.num_transformer_submodules
        if self.num_transformer_submodules is None:
            self.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1

        self.token_dim = config.token_dim

        total_virtual_tokens = self.num_virtual_tokens * self.num_transformer_submodules

        self.prefix_task_cols = Parameter(
            ops.normal(
                mean=0,
                stddev=0.02,
                shape=(self.num_tasks, total_virtual_tokens, self.num_ranks),
            )
        )
        self.prefix_task_rows = Parameter(
            ops.normal(
                mean=0,
                stddev=0.02,
                shape=(self.num_tasks, self.num_ranks, self.token_dim),
            )
        )

        if config.prompt_tuning_init in [
            MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS,
            MultitaskPromptTuningInit.EXACT_SOURCE_TASK,
            MultitaskPromptTuningInit.ONLY_SOURCE_SHARED,
        ]:
            if config.prompt_tuning_init_state_dict_path is None:
                raise ValueError(
                    f"prompt_tuning_init_state_dict_path needs to be specified with {config.prompt_tuning_init} "
                    "init method"
                )

            if config.prompt_tuning_init_state_dict_path.endswith(".safetensors"):
                from mindnlp.utils.serialization import safe_load_file

                state_dict: dict = safe_load_file(config.prompt_tuning_init_state_dict_path)
            elif config.prompt_tuning_init_state_dict_path.endswith(".ckpt"):
                state_dict = mindspore.load_checkpoint(config.prompt_tuning_init_state_dict_path)
            else:
                from mindnlp.utils.serialization import load
                state_dict: dict = load(
                    config.prompt_tuning_init_state_dict_path,
                )

        if config.prompt_tuning_init in [
            MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS,
            MultitaskPromptTuningInit.EXACT_SOURCE_TASK,
        ]:
            prefix_task_cols_: mindspore.Tensor = state_dict["prefix_task_cols"]
            prefix_task_rows_: mindspore.Tensor = state_dict["prefix_task_rows"]

            if config.prompt_tuning_init == MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS:
                prefix_task_cols_ = prefix_task_cols_.mean(0, keep_dims=True)
                prefix_task_rows_ = prefix_task_rows_.mean(0, keep_dims=True)
            elif config.prompt_tuning_init == MultitaskPromptTuningInit.EXACT_SOURCE_TASK:
                prefix_task_cols_ = prefix_task_cols_[config.prompt_tuning_init_task, ...].unsqueeze(0)
                prefix_task_rows_ = prefix_task_rows_[config.prompt_tuning_init_task, ...].unsqueeze(0)

            state_dict = {
                "embedding.weight": state_dict["prompt_embeddings"],
                "prefix_task_cols": prefix_task_cols_,
                "prefix_task_rows": prefix_task_rows_,
            }

            self.load_state_dict(state_dict, strict=True)
        elif config.prompt_tuning_init == MultitaskPromptTuningInit.ONLY_SOURCE_SHARED:
            state_dict = {
                "embedding.weight": state_dict["prompt_embeddings"],
            }

            self.load_state_dict(state_dict, strict=False)

    def construct(self, indices, task_ids):
        if task_ids is None:
            raise ValueError("task_ids cannot be None")

        prompt_embeddings = self.embedding(indices)

        task_cols = ops.index_select(self.prefix_task_cols, 0, task_ids)
        task_rows = ops.index_select(self.prefix_task_rows, 0, task_ids)
        task_prompts = ops.matmul(task_cols, task_rows)

        prompt_embeddings *= task_prompts

        return prompt_embeddings
