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

if is_mindspore_available():
    import mindspore
    from ..models.auto.modeling_auto import (
        MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
    )


class TableQuestionAnsweringArgumentHandler(ArgumentHandler):
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
    """
    Table Question Answering pipeline using a `ModelForTableQuestionAnswering`. This pipeline is only available in
    PyTorch.

    Example:

    ```python
    >>> from mindnlp.transformers import pipeline

    >>> oracle = pipeline(model="google/tapas-base-finetuned-wtq")
    >>> table = {
    ...     "Repository": ["Transformers", "Datasets", "Tokenizers"],
    ...     "Stars": ["36542", "4512", "3934"],
    ...     "Contributors": ["651", "77", "34"],
    ...     "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
    ... }
    >>> oracle(query="How many stars does the transformers repository have?", table=table)
    {'answer': 'AVERAGE > 36542', 'coordinates': [(0, 1)], 'cells': ['36542'], 'aggregator': 'AVERAGE'}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This tabular question answering pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"table-question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a tabular question answering task.
    See the up-to-date list of available models on
    [hf-mirror.com/models](https://hf-mirror.com/models?filter=table-question-answering).
    """

    def __init__(self, args_paser=TableQuestionAnsweringArgumentHandler(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._args_paser = args_paser

        mapping = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES.copy()
        mapping.update(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES)
        self.check_model_type(mapping)

        self.aggregate = bool(getattr(self.model.config, "aggregation_labels", False)) and bool(
            getattr(self.model.config, "aggregation_labels", False)
        )
        self.type = "tape" if hasattr(self.model.config, "aggregation_labels") else None

    def __call__(self, *args, **kwargs):
        r"""
        Answers queries according to a table. The pipeline accepts several types of inputs which are detailed below:

        - `pipeline(table, query)`
        - `pipeline(table, [query])`
        - `pipeline(table=table, query=query)`
        - `pipeline(table=table, query=[query])`
        - `pipeline({"table": table, "query": query})`
        - `pipeline({"table": table, "query": [query]})`
        - `pipeline([{"table": table, "query": query}, {"table": table, "query": query}])`

        The `table` argument should be a dict or a DataFrame built from that dict, containing the whole table:

        Example:

        ```python
        data = {
            "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
            "age": ["56", "45", "59"],
            "number of movies": ["87", "53", "69"],
            "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
        }
        ```

        This dictionary can be passed in as such, or can be converted to a pandas DataFrame:

        Example:

        ```python
        import pandas as pd

        table = pd.DataFrame.from_dict(data)
        ```

        Args:
            table (`pd.DataFrame` or `Dict`):
                Pandas DataFrame or dictionary that will be converted to a DataFrame containing all the table values.
                See above for an example of dictionary.
            query (`str` or `List[str]`):
                Query or list of queries that will be sent to the model alongside the table.
            sequential (`bool`, *optional*, defaults to `False`):
                Whether to do inference sequentially or as a batch. Batching is faster, but models like SQA require the
                inference to be done sequentially to extract relations within sequences, given their conversational
                nature.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Activates and controls padding. Accepts the following values:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).

            truncation (`bool`, `str` or [`TapasTruncationStrategy`], *optional*, defaults to `False`):
                Activates and controls truncation. Accepts the following values:

                - `True` or `'drop_rows_to_fit'`: Truncate to a maximum length specified with the argument `max_length`
                  or to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate row by row, removing rows from the table.
                - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).


        Return:
            A dictionary or a list of dictionaries containing results: Each result is a dictionary with the following
            keys:

            - **answer** (`str`) -- The answer of the query given the table. If there is an aggregator, the answer will
              be preceded by `AGGREGATOR >`.
            - **coordinates** (`List[Tuple[int, int]]`) -- Coordinates of the cells of the answers.
            - **cells** (`List[str]`) -- List of strings made up of the answer cell values.
            - **aggregator** (`str`) -- If the model has an aggregator, this returns the aggregator.
        """
        tqa_pipline_inputs = self._args_paser(*args, **kwargs)

        return super().__call__(tqa_pipline_inputs, **kwargs)

    def _sanitize_parameters(self, sequential=None, padding=None, truncation=None, **kwargs):
        preprocess_param = {}
        if padding is not None:
            preprocess_param["padding"] = padding
        if truncation is not None:
            preprocess_param["truncation"] = truncation

        forword_param = {}
        if sequential is not None:
            forword_param["sequential"] = sequential

        return preprocess_param, forword_param, {}

    def preprocess(self, pipline_input, sequential=None, padding=True, truncation=None):
        if truncation is None:
            truncation = "drop_rows_to_fit" if self.type == "table" \
                else "do_not_truncate"

        table, query = pipline_input["table"], pipline_input["query"]
        if table.empty:
            raise ValueError("table is empty")
        if query is None or query == "":
            raise ValueError("query is empty")

        inputs = self.tokenizer(
            table,
            query,
            truncation=truncation,
            padding=padding,
        )
        inputs["table"] = table
        return inputs

    def _forward(self, model_inputs, sequential=False, **kwargs):
        pass

    def postprocess(self, model_outputs: ModelOutput, **postprocess_parameters: Dict):
        pass
