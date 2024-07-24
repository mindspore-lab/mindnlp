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
"Zero Shot Classification Pipeline"
import inspect
from typing import List, Union

import numpy as np

from mindnlp.utils import logging
from ..tokenization_utils_base import TruncationStrategy
from .base import ArgumentHandler, ChunkPipeline

logger = logging.get_logger(__name__)


class ZeroShotClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for zero-shot for text classification 
    by turning each possible label into an NLI
    premise/hypothesis pair.
    """

    def _parse_labels(self, labels):
        """
        This method '_parse_labels' is a part of the class 'ZeroShotClassificationArgumentHandler' and is responsible
        for parsing and processing the input labels.
        
        Args:
            self (object): The instance of the class.
            labels (str or list): The input labels to be parsed. If a string is provided, it will be split by comma and
                whitespace, and any empty or whitespace-only labels will be removed.
        
        Returns:
            None: This method does not explicitly return any value,
                as it directly modifies the 'labels' parameter in place.
        
        Raises:
            None.
        """
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",") if label.strip()]
        return labels

    def __call__(self, sequences, labels, hypothesis_template):
        """
        Class: ZeroShotClassificationArgumentHandler
        
        Method: __call__
        
        Description:
            This method processes the input sequences, labels, and hypothesis template to generate
            sequence pairs for zero-shot classification.

        Args:
            self (object): The instance of the class.
            sequences (str or list): The input sequences to be classified.
                If a string is provided, it will be converted to a list with the string as the only element.
            labels (list): The list of target labels for classification.
            hypothesis_template (str): The template string used to format the hypothesis for each label.

        Returns:
            None: This method does not return any value. It updates the instance with the generated sequence pairs.

        Raises:
            ValueError: Raised if either the 'labels' or 'sequences' parameter is empty.
            ValueError: Raised if the 'hypothesis_template' cannot be formatted with the target labels.
            TypeError: Raised if the 'sequences' parameter is not a string or a list.
        """
        if len(labels) == 0 or len(sequences) == 0:
            raise ValueError("You must include at least one label and at least one sequence.")
        if hypothesis_template.format(labels[0]) == hypothesis_template:
            raise ValueError(
                (
                    f'The provided hypothesis_template "{hypothesis_template}" was not able to be formatted with the target labels. '
                    f'Make sure the passed template includes formatting syntax such as {{}} where the label should go.'
                )
            )

        if isinstance(sequences, str):
            sequences = [sequences]

        sequence_pairs = []
        for sequence in sequences:
            sequence_pairs.extend([[sequence, hypothesis_template.format(label)] for label in labels])

        return sequence_pairs, sequences


class ZeroShotClassificationPipeline(ChunkPipeline):
    """
    NLI-based zero-shot classification pipeline using a `ModelForSequenceClassification` trained on NLI (natural
    language inference) tasks. Equivalent of `text-classification` pipelines, but these models don't require a
    hardcoded number of potential classes, they can be chosen at runtime. It usually means it's slower but it is
    **much** more flexible.

    Any combination of sequences and labels can be passed and each combination will be posed as a premise/hypothesis
    pair and passed to the pretrained model. Then, the logit for *entailment* is taken as the logit for the candidate
    label being valid. Any NLI model can be used, but the id of the *entailment* label must be included in the model
    config's :attr:*~transformers.PretrainedConfig.label2id*.

    Example:
        ```python
        >>> from transformers import pipeline
        ...
        >>> oracle = pipeline(model="facebook/bart-large-mnli")
        >>> oracle(
        ...     "I have a problem with my iphone that needs to be resolved asap!!",
        ...     candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
        ... )
        {'sequence': 'I have a problem with my iphone that needs to be resolved asap!!',
            'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'],
            'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}
        ...
        >>> oracle(
        ...     "I have a problem with my iphone that needs to be resolved asap!!",
        ...     candidate_labels=["english", "german"],
        ... )
        {'sequence': 'I have a problem with my iphone that needs to be resolved asap!!',
            'labels': ['english', 'german'], 'scores': [0.814, 0.186]}
        ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This NLI pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-classification"`.

    The models that this pipeline can use are models that have been fine-tuned on an NLI task.
    See the up-to-date list
    of available models on [hf-mirror.com/models](https://hf-mirror.com/models?search=nli).
    """

    def __init__(self, *args, args_parser=ZeroShotClassificationArgumentHandler(), **kwargs):
        """
        Initializes a new instance of the ZeroShotClassificationPipeline class.

        Args:
            self: The instance of the ZeroShotClassificationPipeline class.
            *args: Variable length argument list.
            args_parser:
                An instance of the ZeroShotClassificationArgumentHandler class that handles the arguments for
                zero-shot classification. Defaults to ZeroShotClassificationArgumentHandler().
            **kwargs: Keyword arguments.

        Returns:
            None.

        Raises:
            None.
        """
        self._args_parser = args_parser
        super().__init__(*args, **kwargs)
        if self.entailment_id == -1:
            logger.warning(
                "Failed to determine 'entailment' label id from the label2id mapping in the model config. Setting to "
                "-1. Define a descriptive label2id mapping in the model config to ensure correct outputs."
            )

    @property
    def entailment_id(self):
        """
        Returns the index of the 'entailment' label in the label-to-identifier mapping of the
        ZeroShotClassificationPipeline's model configuration.

        Args:
            self (ZeroShotClassificationPipeline): The current instance of the ZeroShotClassificationPipeline class.

        Returns:
            int: The index of the 'entailment' label in the label-to-identifier mapping. If the 'entailment' label is
                not found, -1 is returned.

        Raises:
            None.

        """
        for label, ind in self.model.config.label2id.items():
            if label.lower().startswith("entail"):
                return ind
        return -1

    def _parse_and_tokenize(
            self, sequence_pairs, padding=True, add_special_tokens=True, truncation=TruncationStrategy.ONLY_FIRST
    ):
        """
        Parse arguments and tokenize only_first so that hypothesis (label) is not truncated
        """
        return_tensors = 'ms'
        if self.tokenizer.pad_token is None:
            # Override for tokenizers not supporting padding
            logger.error(
                "Tokenizer was not supporting padding necessary for zero-shot, attempting to use "
                " `pad_token=eos_token`"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            inputs = self.tokenizer(
                sequence_pairs,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
            )
        except Exception as exception:
            if "too short" in str(exception):
                # tokenizers might yell that we want to truncate
                # to a value that is not even reached by the input.
                # In that case we don't want to truncate.
                # It seems there's not a really better way to catch that
                # exception.

                inputs = self.tokenizer(
                    sequence_pairs,
                    add_special_tokens=add_special_tokens,
                    return_tensors=return_tensors,
                    padding=padding,
                    truncation=TruncationStrategy.DO_NOT_TRUNCATE,
                )
            else:
                raise exception

        return inputs

    def _sanitize_parameters(self, **kwargs):
        """
        Sanitizes the parameters for the ZeroShotClassificationPipeline.

        Args:
            self: An instance of the ZeroShotClassificationPipeline class.

        Returns:
            None.

        Raises:
            None.

        This method performs the following tasks:

        - Renames the deprecated 'multi_class' argument to 'multi_label' if provided and logs a warning.
        - Parses and sanitizes the 'candidate_labels' parameter if provided.
        - Retrieves the 'hypothesis_template' parameter if provided.
        - Collects the 'multi_label' parameter if provided.

        Note:
            - The 'multi_class' argument has been deprecated and renamed to 'multi_label'. 'multi_class' will be
            removed in a future version of Transformers.
            - The 'candidate_labels' parameter should be a list of strings representing the labels.
            - The 'hypothesis_template' parameter should be a string representing the template for the hypothesis.
            - The 'multi_label' parameter should be a boolean indicating whether multi-label classification should be used.

        Example:
            ```python
            >>> pipeline = ZeroShotClassificationPipeline()
            >>> pipeline._sanitize_parameters(multi_class=True, candidate_labels=['label1', 'label2'], hypothesis_template='This text is about {}.')
            ```
        """
        if kwargs.get("multi_class", None) is not None:
            kwargs["multi_label"] = kwargs["multi_class"]
            logger.warning(
                "The `multi_class` argument has been deprecated and renamed to `multi_label`. "
                "`multi_class` will be removed in a future version of Transformers."
            )
        preprocess_params = {}
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = self._args_parser._parse_labels(kwargs["candidate_labels"])
        if "hypothesis_template" in kwargs:
            preprocess_params["hypothesis_template"] = kwargs["hypothesis_template"]

        postprocess_params = {}
        if "multi_label" in kwargs:
            postprocess_params["multi_label"] = kwargs["multi_label"]
        return preprocess_params, {}, postprocess_params

    def __call__(
            self,
            sequences: Union[str, List[str]],
            *args,
            **kwargs,
    ):
        """
        Classify the sequence(s) given as inputs. See the [`ZeroShotClassificationPipeline`] documentation
        for more information.

        Args:
            sequences (`str` or `List[str]`):
                The sequence(s) to classify, will be truncated if the model input is too large.
            candidate_labels (`str` or `List[str]`):
                The set of possible class labels to classify each sequence into.
                Can be a single label, a string of
                comma-separated labels, or a list of labels.
            hypothesis_template (`str`, *optional*, defaults to `"This example is {}."`):
                The template used to turn each label into an NLI-style hypothesis.
                This template must include a {} or similar syntax for the candidate
                label to be inserted into the template. For example, the default
                template is `"This example is {}."` With the candidate label `"sports"`,
                this would be fed into the model like `"<cls> sequence to classify <sep> This example is sports . <sep>"`.
                The default templateworks well in many cases,
                but it may be worthwhile to experiment with different templates depending on the task setting.
            multi_label (`bool`, *optional*, defaults to `False`):
                Whether or not multiple candidate labels can be true. If `False`, the scores are normalized such that
                the sum of the label likelihoods for each sequence is 1. If `True`,
                the labels are considered independent and probabilities are normalized for each candidate
                by doing a softmax of the entailment score vs. the contradiction score.

        Returns:
            A `dict` or a list of `dict`:
                Each result comes as a dictionary with the following keys:

                - **sequence** (`str`) -- The sequence for which this is the output.
                - **labels** (`List[str]`) -- The labels sorted by order of likelihood.
                - **scores** (`List[float]`) -- The probabilities for each of the labels.
        """
        if len(args) == 0:
            pass
        elif len(args) == 1 and "candidate_labels" not in kwargs:
            kwargs["candidate_labels"] = args[0]
        else:
            raise ValueError(f"Unable to understand extra arguments {args}")

        return super().__call__(sequences, **kwargs)

    def preprocess(self, inputs, candidate_labels=None, hypothesis_template="This example is {}."):
        """
        This method preprocesses inputs for zero-shot classification and generates model inputs for each candidate label.

        Args:
            self: The instance of the ZeroShotClassificationPipeline class.
            inputs: The input sequences to be classified.
            candidate_labels: The list of candidate labels for classification. Defaults to None.
            hypothesis_template: The template string for the hypothesis. Defaults to 'This example is {}'.

        Returns:
            None: This method yields dictionaries with model inputs for each candidate label.

        Raises:
            None.
        """
        sequence_pairs, sequences = self._args_parser(inputs, candidate_labels, hypothesis_template)
        for i, (candidate_label, sequence_pair) in enumerate(zip(candidate_labels, sequence_pairs)):
            model_input = self._parse_and_tokenize([sequence_pair])
            yield {
                "candidate_label": candidate_label,
                "sequence": sequences[0],
                "is_last": i == len(candidate_labels) - 1,
                **model_input,
            }

    def _forward(self, inputs):
        """
        Executes the forward pass for the ZeroShotClassificationPipeline.

        Args:
            self (ZeroShotClassificationPipeline): The instance of the ZeroShotClassificationPipeline class.
            inputs (dict): A dictionary containing the input data for the forward pass.
                - candidate_label (str): The candidate label for classification.
                - sequence (str): The sequence to classify.

        Returns:
            None

        Raises:
            None
        """
        candidate_label = inputs["candidate_label"]
        sequence = inputs["sequence"]
        model_inputs = {k: inputs[k] for k in self.tokenizer.model_input_names}

        #`XXForSequenceClassification` models should not use `use_cache=True` even if it's supported
        model_forward = self.model.forward
        if "use_cache" in inspect.signature(model_forward).parameters.keys():
            model_inputs["use_cache"] = False
        outputs = model_forward(**model_inputs)
        model_outputs = {
            "candidate_label": candidate_label,
            "sequence": sequence,
            "is_last": inputs["is_last"],
            **outputs,
        }
        return model_outputs

    def postprocess(self, model_outputs, multi_label=False):
        """
        This method postprocesses the model outputs for a ZeroShotClassificationPipeline.

        Args:
            self (object): The instance of the ZeroShotClassificationPipeline class.
            model_outputs (list): A list of dictionaries containing the model outputs.
                Each dictionary must have the keys 'candidate_label', 'sequence', and 'logits'.
                The 'candidate_label' key represents the candidate label, 'sequence' key represents the sequence,
                and 'logits' key holds the logits values.
            multi_label (bool): A flag indicating whether the classification is multi-label or not.
                If set to True, the method processes the outputs accordingly.

        Returns:
            dict: A dictionary containing the processed information of the model outputs.
                It includes the 'sequence' key with the sequence value, 'labels' key with the list of candidate labels
                in descending order of their scores, and 'scores' key with the corresponding scores of the candidate
                labels.
        
        Raises:
            IndexError: If the indices accessed during processing are out of bounds.
            ValueError: If there are issues with the input data or calculations within the method.
        """
        candidate_labels = [outputs["candidate_label"] for outputs in model_outputs]
        sequences = [outputs["sequence"] for outputs in model_outputs]
        logits = np.concatenate([output["logits"].numpy() for output in model_outputs])
        num_examples = logits.shape[0]
        num_candidates = len(candidate_labels)
        num_sequences = num_examples // num_candidates
        reshaped_outputs = logits.reshape((num_sequences, num_candidates, -1))

        if multi_label or len(candidate_labels) == 1:
            # softmax over the entailment vs. contradiction dim for each label independently
            entailment_id = self.entailment_id
            contradiction_id = -1 if entailment_id == 0 else 0
            entail_contr_logits = reshaped_outputs[..., [contradiction_id, entailment_id]]
            scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(-1, keepdims=True)
            scores = scores[..., 1]
        else:
            # softmax the "entailment" logits over all candidate labels
            entail_logits = reshaped_outputs[..., self.entailment_id]
            scores = np.exp(entail_logits) / np.exp(entail_logits).sum(-1, keepdims=True)

        top_inds = list(reversed(scores[0].argsort()))
        return {
            "sequence": sequences[0],
            "labels": [candidate_labels[i] for i in top_inds],
            "scores": scores[0, top_inds].tolist()}
