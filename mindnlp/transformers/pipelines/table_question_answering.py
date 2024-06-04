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
"""table question answering pipeline."""

import types
from typing import Any, Dict

from .base import ArgumentHandler, Dataset, GenericTensor
from .. import Pipeline
from ...utils import (
    is_mindspore_available,
    is_tokenizers_available,
    requires_backends, ModelOutput
)


class TableQuestionAnsweringArguementHandler(ArgumentHandler):
    """
    Handles arguments for the TableQuestionAnsweringPipeline
    """

    def __call__(self, table=None, query=None, **kwargs):
        """
        Returns tqa_pipeline_inputs of shape:
        [
            {"table": pd.DataFrame, "query": List[str]},
            ...,
            {"table": pd.DataFrame, "query" : List[str]}
        ]
        Args:
            table:
            query:

        Returns:

        """
        import pandas as pd

        if table is None:
            raise ValueError("Keyword argument `table` cannot be None.")
        elif query is None:
            if isinstance(table, dict) and table.get("table") is not None and table.get("query") is not None:
                tqa_pipeline_inputs = [table]
            elif isinstance(table, list) and len(table) > 0:
                if not all(isinstance(d, dict) for d in table):
                    raise ValueError(
                        f"Keyword argument `table` should be a list of dict, but is {(type(d) for d in table)}"
                    )

                if table[0].get("query") is not None and table[0].get("table") is not None:
                    tqa_pipeline_inputs = table
                else:
                    raise ValueError(
                        "If keyword argument `table` is a list of dictionaries, each dictionary should have a `table`"
                        f" and `query` key, but only dictionary has keys {table[0].keys()} `table` and `query` keys."
                    )
            elif Dataset is not None and isinstance(table, Dataset) or isinstance(table, types.GeneratorType):
                return table
            else:
                raise ValueError(
                    "Invalid input. Keyword argument `table` should be either of type `dict` or `list`, but "
                    f"is {type(table)})"
                )
        else:
            tqa_pipeline_inputs = [{"table": table, "query": query}]

        for tqa_pipeline_input in tqa_pipeline_inputs:
            if not isinstance(tqa_pipeline_input["table"], pd.DataFrame):
                if tqa_pipeline_input["table"] is None:
                    raise ValueError("Table cannot be None.")

                tqa_pipeline_input["table"] = pd.DataFrame(tqa_pipeline_input["table"])

        return tqa_pipeline_inputs


class TableQuestionAnsweringPipeline(Pipeline):
    def postprocess(self, model_outputs: ModelOutput, **postprocess_parameters: Dict) -> Any:
        pass

    def _forward(self, input_tensors: Dict[str, GenericTensor], **forward_parameters: Dict) -> ModelOutput:
        pass

    def preprocess(self, input_: Any, **preprocess_parameters: Dict) -> Dict[str, GenericTensor]:
        pass

    def _sanitize_parameters(self, **pipeline_parameters):
        pass
