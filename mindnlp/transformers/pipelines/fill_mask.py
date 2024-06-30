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
"Fill Mask Pipeline"
from typing import Dict

import numpy as np

from mindnlp.utils import is_mindspore_available,logging
from .base import GenericTensor, Pipeline, PipelineException


if is_mindspore_available():
    from mindspore import ops


logger = logging.get_logger(__name__)


class FillMaskPipeline(Pipeline):
    """
    Masked language modeling prediction pipeline using any `ModelWithLMHead`.
    See the [masked language modeling
    examples](../task_summary#masked-language-modeling) for more information.

    Example:
        ```python
        >>> from mindnlp.transformers import pipeline
        ...
        >>> fill_masker = pipeline(model="google-bert/bert-base-uncased")
        >>> fill_masker("This is a simple [MASK].")
        [{'score': 0.042, 'token': 3291, 'token_str': 'problem',
        'sequence': 'this is a simple problem.'},
        {'score': 0.031, 'token': 3160, 'token_str': 'question',
        'sequence': 'this is a simple question.'},
        {'score': 0.03, 'token': 8522, 'token_str': 'equation',
        'sequence': 'this is a simple equation.'},
        {'score': 0.027, 'token': 2028, 'token_str': 'one', 'sequence': 'this is a simple one.'},
        {'score': 0.024, 'token': 3627, 'token_str': 'rule', 'sequence': 'this is a simple rule.'}]
        ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This mask filling pipeline can currently be loaded from [`pipeline`]
    using the following task identifier:
    `"fill-mask"`.

    The models that this pipeline can use are models
    that have been trained with a masked language modeling objective,
    which includes the bi-directional models in the library.
    See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=fill-mask).

    <Tip>

    This pipeline only works for inputs with exactly one token masked.
    Experimental: We added support for multiple
    masks. The returned values are raw model output,
    and correspond to disjoint probabilities where one might expect
    joint probabilities (See [discussion](https://github.com/huggingface/transformers/pull/10222)).

    </Tip>

    <Tip>

    This pipeline now supports tokenizer_kwargs.

    Example:
        ```python
        >>> from mindnlp.transformers import pipeline
        ...
        >>> fill_masker = pipeline(model="google-bert/bert-base-uncased")
        >>> tokenizer_kwargs = {"truncation": True}
        >>> fill_masker(
        ...     "This is a simple [MASK]. " + "...with a large amount of repeated text appended. " * 100,
        ...     tokenizer_kwargs=tokenizer_kwargs,
        ... )
        ```

    </Tip>
    """
    def get_masked_index(self, input_ids: GenericTensor) -> np.ndarray:
        """
        This method returns the indices of the masked tokens in the input tensor.

        Args:
            self (FillMaskPipeline): The instance of the FillMaskPipeline class.
            input_ids (GenericTensor): The input tensor containing token IDs. It should be compatible with the
                operations performed by the method.

        Returns:
            np.ndarray: An array of indices representing the positions of the masked tokens in the input tensor.

        Raises:
            None
        """
        masked_index = ops.nonzero(input_ids == self.tokenizer.mask_token_id)
        return masked_index

    def _ensure_exactly_one_mask_token(self, input_ids: GenericTensor) -> np.ndarray:
        """
        Ensure that there is exactly one mask token in the input and return the masked index as a NumPy array.

        Args:
            self: An instance of the FillMaskPipeline class.
            input_ids (GenericTensor): The input tensor containing tokens.
                It is used to identify the mask token in the input.
                Should be a 2D tensor with shape (batch_size, sequence_length).

        Returns:
            np.ndarray: A NumPy array representing the masked index.
                The array contains the index of the mask token in the input tensor.

        Raises:
            PipelineException: If no mask token is found in the input tensor.
                This exception is raised with the context of 'fill-mask' operation,
                the model's base model prefix, and the missing mask token.
        """
        masked_index = self.get_masked_index(input_ids)
        numel = np.prod(masked_index.shape)
        if numel < 1:
            raise PipelineException(
                "fill-mask",
                self.model.base_model_prefix,
                f"No mask_token ({self.tokenizer.mask_token}) found on the input",
            )

    def ensure_exactly_one_mask_token(self, model_inputs: GenericTensor):
        """
        Ensure that there is exactly one mask token in the input tensor(s) provided to the FillMaskPipeline.

        Args:
            self: An instance of the FillMaskPipeline class.
            model_inputs (GenericTensor): The input tensor(s) to the model.
                It can be either a single tensor or a list of tensors. Each tensor should have an 'input_ids' field.

        Returns:
            None.

        Raises:
            None.

        This method iterates through the input tensor(s) and checks if there is exactly one mask token present.
        If the 'model_inputs' parameter is a list, it iterates through each tensor in the list and ensures that the
        first 'input_ids' tensor has exactly one mask token. If 'model_inputs' is not a list, it assumes it is a single
        tensor and checks each 'input_ids' tensor to ensure that it has exactly one mask token.
        """
        if isinstance(model_inputs, list):
            for model_input in model_inputs:
                self._ensure_exactly_one_mask_token(model_input["input_ids"][0])
        else:
            for input_ids in model_inputs["input_ids"]:
                self._ensure_exactly_one_mask_token(input_ids)

    def preprocess(
        self, inputs, return_tensors=None, tokenizer_kwargs=None, **preprocess_parameters
    ) -> Dict[str, GenericTensor]:
        """
        This method preprocesses the inputs using the tokenizer and returns the preprocessed model inputs.

        Args:
            self: The instance of the FillMaskPipeline class.
            inputs: The input data to be preprocessed.
            return_tensors: (Optional) Specifies the desired format of the returned tensors. Default is 'ms'.
                Allowed values are 'ms' (for model-specific tensors) or 'pt' (for PyTorch tensors).
            tokenizer_kwargs: (Optional) Additional keyword arguments to be passed to the tokenizer.

        Returns:
            Dict[str, GenericTensor]:
                A dictionary containing the preprocessed model inputs, with keys representing the input types
                and values representing the corresponding GenericTensor objects.

        Raises:
            None.
        """
        if return_tensors is None:
            return_tensors = 'ms'
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        model_inputs = self.tokenizer(inputs, return_tensors=return_tensors, **tokenizer_kwargs)
        self.ensure_exactly_one_mask_token(model_inputs)
        return model_inputs

    def _forward(self, model_inputs):
        """
        Method _forward in the FillMaskPipeline class.

        Args:
            self (FillMaskPipeline): The instance of the FillMaskPipeline class.
            model_inputs (dict): A dictionary containing the inputs required by the model.
                It should include key-value pairs for the input_ids that the model expects.

        Returns:
            None.

        Raises:
            None.
        """
        model_outputs = self.model(**model_inputs)
        model_outputs["input_ids"] = model_inputs["input_ids"]
        return model_outputs

    def postprocess(self, model_outputs, top_k=5, target_ids=None):
        '''
        Method: postprocess

        This method takes 4 parameters: self, model_outputs, top_k, target_ids.

        Args:
            self (FillMaskPipeline): The instance of the FillMaskPipeline class.
            model_outputs (dict): The dictionary containing the model outputs, including 'input_ids' and 'logits'.
            top_k (int): The maximum number of top predictions to consider. Defaults to 5.
            target_ids (Tensor, optional): The tensor containing the target token IDs.
                - If provided, only predictions for these target token IDs will be considered.
                - If not provided, all token IDs will be considered.

        Returns:
            None.

        Raises:
            ValueError: If target_ids is provided and its shape is less than top_k.
            IndexError: If masked_index is out of range.
            TypeError: If the input types are not as expected.
        '''
        # Cap top_k if there are targets
        if target_ids is not None and target_ids.shape[0] < top_k:
            top_k = target_ids.shape[0]
        input_ids = model_outputs["input_ids"][0]
        outputs = model_outputs["logits"]

        masked_index = ops.nonzero(input_ids == self.tokenizer.mask_token_id).squeeze(-1)
        # Fill mask pipeline supports only one ${mask_token} per sample

        logits = outputs[0, masked_index, :]
        probs=ops.softmax(logits,axis=-1)
        if target_ids is not None:
            target_ids=list(target_ids)
            probs = probs[..., target_ids]

        values, predictions = probs.topk(top_k)

        result = []
        single_mask = values.shape[0] == 1
        for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
            row = []
            for v, p in zip(_values, _predictions):
                # Copy is important since we're going to modify this array in place
                tokens = input_ids.numpy().copy()
                if target_ids is not None:
                    p = target_ids[p].tolist()

                tokens[masked_index[i]] = p
                # Filter padding out:
                tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
                # Originally we skip special tokens to give readable output.
                # For multi masks though, the other [MASK] would be removed otherwise
                # making the output look odd, so we add them back
                sequence = self.tokenizer.decode(tokens, skip_special_tokens=single_mask)
                proposition = {"score": v, "token": p,
                               "token_str": self.tokenizer.decode([p]), "sequence": sequence}
                row.append(proposition)
            result.append(row)
        if single_mask:
            return result[0]
        return result

    def get_target_ids(self, targets, top_k=None):
        """
        Method: get_target_ids

        Returns a list of target token IDs from the model vocabulary for the specified targets.

        Args:
            self (FillMaskPipeline): An instance of the FillMaskPipeline class.
            targets (str or List[str]): A string or a list of strings representing the target tokens.
            top_k (int, optional): The maximum number of target IDs to return. Defaults to None.

        Returns:
            target_ids (list): A list of unique target token IDs from the model vocabulary.

        Raises:
            ValueError: If no target is provided.
            Any Exception: If an error occurs while retrieving the model vocabulary.

        Note:
            - If a single target string is passed, it will be converted into a list containing that string.
            - If a target token does not exist in the model vocabulary, it will be replaced with a meaningful
            token if possible.
            - If a target token does not exist in the model vocabulary and cannot be replaced, it will be ignored and
            a warning will be logged.

        Example:
            ```python
            >>> pipeline = FillMaskPipeline()
            >>> targets = ['apple', 'banana', 'orange']
            >>> result = pipeline.get_target_ids(targets, top_k=2)
            >>> print(result)
            [135, 742]
            ```
        """
        if isinstance(targets, str):
            targets = [targets]
        try:
            vocab = self.tokenizer.get_vocab()
        except Exception:
            vocab = {}
        target_ids = []
        for target in targets:
            id_ = vocab.get(target, None)
            if id_ is None:
                input_ids = self.tokenizer(
                    target,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    max_length=1,
                    truncation=True,
                )["input_ids"]
                if len(input_ids) == 0:
                    logger.warning(
                        f"The specified target token `{target}` "
                        f"does not exist in the model vocabulary. "
                        "We cannot replace it with anything meaningful, ignoring it"
                    )
                    continue
                id_ = input_ids[0]
                # it becomes pretty slow, so let's make sure
                # The warning enables them to fix the input to
                # get faster performance.
                logger.warning(
                    f"The specified target token `{target}` does not exist "
                    f"in the model vocabulary. "
                    f"Replacing with `{self.tokenizer.convert_ids_to_tokens(id_)}`."
                )
            target_ids.append(id_)
        target_ids = list(set(target_ids))
        if len(target_ids) == 0:
            raise ValueError("At least one target must be provided when passed.")
        target_ids = np.array(target_ids)
        return target_ids

    def _sanitize_parameters(self, top_k=None, targets=None, tokenizer_kwargs=None):
        """
        This method '_sanitize_parameters' is defined in the class 'FillMaskPipeline'.
        It is responsible for sanitizing the input parameters for the FillMaskPipeline.

        Args:
            self (object): The instance of the FillMaskPipeline class.
            top_k (int, optional): The maximum number of predictions to return. Defaults to None.
            targets (str, optional): The target words or phrases for the fill-mask task. Defaults to None.
            tokenizer_kwargs (dict, optional): Additional keyword arguments for the tokenizer. Defaults to None.

        Returns:
            tuple: A tuple containing preprocess_params, an empty dict, and postprocess_params.
                preprocess_params may contain 'tokenizer_kwargs' if provided,
                and postprocess_params may contain 'target_ids' and 'top_k' if the corresponding arguments are provided.

        Raises:
            PipelineException: Raised if the tokenizer does not define a `mask_token`.
        """
        preprocess_params = {}

        if tokenizer_kwargs is not None:
            preprocess_params["tokenizer_kwargs"] = tokenizer_kwargs

        postprocess_params = {}

        if targets is not None:
            target_ids = self.get_target_ids(targets, top_k)
            postprocess_params["target_ids"] = target_ids

        if top_k is not None:
            postprocess_params["top_k"] = top_k

        if self.tokenizer.mask_token_id is None:
            raise PipelineException(
                "fill-mask", self.model.base_model_prefix, "The tokenizer does not define a `mask_token`."
            )
        return preprocess_params, {}, postprocess_params

    def __call__(self, inputs, *args, **kwargs):
        """
        Fill the masked token in the text(s) given as inputs.

        Args:
            args (`str` or `List[str]`):
                One or several texts (or one list of prompts) with masked tokens.
            targets (`str` or `List[str]`, *optional*):
                When passed, the model will limit the scores to the passed targets instead of looking up in the whole
                vocab. If the provided targets are not in the model vocab, they will be tokenized and the first
                resulting token will be used (with a warning, and that might be slower).
            top_k (`int`, *optional*):
                When passed, overrides the number of predictions to return.

        Returns:
            A list or a list of list of `dict`:
                Each result comes as list of dictionaries with the following keys:

                - **sequence** (`str`) -- The corresponding input with the mask token prediction.
                - **score** (`float`) -- The corresponding probability.
                - **token** (`int`) -- The predicted token id (to replace the masked one).
                - **token_str** (`str`) -- The predicted token (to replace the masked one).
        """
        outputs = super().__call__(inputs, **kwargs)
        if isinstance(inputs, list) and len(inputs) == 1:
            return outputs[0]
        return outputs
