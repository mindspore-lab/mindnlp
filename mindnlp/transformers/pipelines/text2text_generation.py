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
""" Text2Text Generation Pipeline"""
import enum
import warnings
from mindnlp.utils import logging
from .base import Pipeline

from ..tokenization_utils_base import TruncationStrategy

logger = logging.get_logger(__name__)


class ReturnType(enum.Enum):

    """
    Represents the different return types for a function in Python.
    
    This class defines different return types that can be associated with a function's output.
    It inherits from enum.Enum, allowing the return types to be accessed as members of the ReturnType class.
    
    Attributes:
        SUCCESS: Represents a successful return from a function.
        FAILURE: Represents a failed return from a function.
        ERROR: Represents an error occurred during the function execution.
    
    Example:
        To access the SUCCESS return type:

        - ReturnType.SUCCESS

        To check if a return type is of type FAILURE:

        - ReturnType.FAILURE == ReturnType.FAILURE
    """
    TENSORS = 0
    TEXT = 1


class Text2TextGenerationPipeline(Pipeline):
    """
    Pipeline for text to text generation using seq2seq models.

    Example:
        ```python
        >>> from mindnlp.transformers import pipeline
        ...
        >>> generator = pipeline("text2text-generation", model="t5-base")
        >>> generator(
        ...     "answer: Manuel context: Manuel has created RuPERTa-base with the support of HF-Transformers and Google"
        ... )
        [{'generated_text': 'question: Who created the RuPERTa-base?'}]
        ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial). You can pass text
    generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about
    text generation parameters in [Text generation strategies](../generation_strategies) and [Text
    generation](text_generation).

    This Text2TextGenerationPipeline pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"text2text-generation"`.

    The models that this pipeline can use are models that have been fine-tuned on a translation task. See the
    up-to-date list of available models on
    [hf-mirror.com/models](https://hf-mirror.com/models?filter=text2text-generation). For a list of available
    parameters, see the [following
    documentation](https://hf-mirror.com/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

    Example:
        ```python
        >>> text2text_generator = pipeline("text2text-generation")
        >>> text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")
        ```
    """
    # Used in the return key of the pipeline.
    return_name = "generated"

    def __init__(self, *args, **kwargs):
        """
        Initializes an instance of Text2TextGenerationPipeline.

        Args:
            self: An instance of the Text2TextGenerationPipeline class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(
            self,
            return_tensors=None,
            return_text=None,
            return_type=None,
            clean_up_tokenization_spaces=None,
            truncation=None,
            stop_sequence=None,
            **generate_kwargs,
    ):
        """
        This method '_sanitize_parameters' in the 'Text2TextGenerationPipeline' class sanitizes and prepares input
        parameters for text generation.

        Args:
            self: The instance of the class.
            return_tensors (bool): Whether to return the generated text as tensors. Default is None.
            return_text (bool): Whether to return the generated text as plain text. Default is None.
            return_type (ReturnType): The type of output to return. Can be either ReturnType.TENSORS or ReturnType.TEXT.
                Default is None.
            clean_up_tokenization_spaces (bool): Whether to clean up tokenization spaces in the generated text.
                Default is None.
            truncation (bool): Whether to truncate the generated text. Default is None.
            stop_sequence (str): The sequence of tokens that indicates the end of generation.

        Returns:
            preprocess_params (dict): A dictionary containing preprocessing parameters.
            forward_params (dict): A dictionary containing forward pass parameters.
            postprocess_params (dict): A dictionary containing postprocessing parameters.

        Raises:
            Warning: When attempting to stop generation on a multiple token sequence, as this feature is not
                supported in transformers.
        """
        preprocess_params = {}
        if truncation is not None:
            preprocess_params["truncation"] = truncation

        forward_params = generate_kwargs

        postprocess_params = {}
        if return_tensors is not None and return_type is None:
            return_type = ReturnType.TENSORS if return_tensors else ReturnType.TEXT
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces
        if return_text is not None:
            postprocess_params["return_type"] = ReturnType.TEXT if return_text else ReturnType.TENSORS

        if stop_sequence is not None:
            stop_sequence_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
            if len(stop_sequence_ids) > 1:
                warnings.warn(
                    "Stopping on a multiple token sequence is not yet supported on transformers. The first token of"
                    " the stop sequence will be used as the stop sequence string in the interim."
                )
            generate_kwargs["eos_token_id"] = stop_sequence_ids[0]

        return preprocess_params, forward_params, postprocess_params

    def check_inputs(self, input_length: int, min_length: int, max_length: int):
        """
        Checks whether there might be something wrong with given input with regard to the model.
        """
        if input_length < min_length:
            logger.warning(
                f"Your min_length is set to {min_length}, but you input_length is only {input_length}. You might "
                "consider decreasing min_length manually, e.g. summarizer('...', min_length=10)"
            )
        if input_length > max_length:
            logger.warning(
                f"Your max_length is set to {max_length}, but you input_length is only {input_length}. You might "
                "consider increasing max_length manually, e.g. summarizer('...', max_length=400)"
            )

        return True

    def _parse_and_tokenize(self, *args, truncation):
        """
        This method '_parse_and_tokenize' is a member of the class 'Text2TextGenerationPipeline'.
        It parses and tokenizes input text data using a tokenizer and prepares it for model inference.

        Args:
            self: An instance of the Text2TextGenerationPipeline class.

        Returns:
            None.

        Raises:
            ValueError:
                Raised if the tokenizer's pad_token_id is not set when batch input is used or if the input data
                format is incorrect.
        """
        prefix = self.model.config.prefix if self.model.config.prefix is not None else ""
        if isinstance(args[0], list):
            if self.tokenizer.pad_token_id is None:
                raise ValueError("Please make sure that the tokenizer has a pad_token_id when using a batch input")
            args = ([prefix + arg for arg in args[0]],)
            padding = True

        elif isinstance(args[0], str):
            args = (prefix + args[0],)
            padding = False
        else:
            raise ValueError(
                f" `args[0]`: {args[0]} have the wrong format. The should be either of type `str` or type `list`"
            )
        inputs = self.tokenizer(*args, padding=padding, truncation=truncation, return_tensors='ms')
        # This is produced by tokenizers but is an invalid generate kwargs
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        return inputs

    def __call__(self, *args, **kwargs):
        r"""
        Generate the output text(s) using text(s) given as inputs.

        Args:
            args (`str` or `List[str]`):
                Input text for the encoder.
            return_tensors (`bool`, *optional*, defaults to `False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (`bool`, *optional*, defaults to `True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the potential extra spaces in the text output.
            truncation (`TruncationStrategy`, *optional*, defaults to `TruncationStrategy.DO_NOT_TRUNCATE`):
                The truncation strategy for the tokenization within the pipeline. `TruncationStrategy.DO_NOT_TRUNCATE`
                (default) will never truncate, but it is sometimes desirable to truncate the input to fit the model's
                max_length instead of throwing an error down the line.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).

        Returns:
            A list or a list of list of `dict`:
                Each result comes as a dictionary with the following keys:

                - **generated_text** (`str`, present when `return_text=True`) -- The generated text.
                - **generated_token_ids** (`torch.Tensor` or `tf.Tensor`, present when `return_tensors=True`) -- The token
                  ids of the generated text.
        """
        result = super().__call__(*args, **kwargs)
        if (
                isinstance(args[0], list)
                and all(isinstance(el, str) for el in args[0])
                and all(len(res) == 1 for res in result)
        ):
            return [res[0] for res in result]
        return result

    def preprocess(self, inputs, truncation=TruncationStrategy.DO_NOT_TRUNCATE, **kwargs):
        """
        Preprocesses the input text for text-to-text generation.

        Args:
            self (Text2TextGenerationPipeline): The instance of the Text2TextGenerationPipeline class.
            inputs (Union[str, List[str]]): The input text or list of input texts to be preprocessed.
            truncation (TruncationStrategy, optional): The strategy to use for truncating the input text
                if it exceeds the maximum length. Defaults to TruncationStrategy.DO_NOT_TRUNCATE.
            **kwargs: Additional keyword arguments to be passed to the _parse_and_tokenize method.

        Returns:
            None.

        Raises:
            TypeError: If the inputs parameter is not a string or a list of strings.
            ValueError: If the truncation parameter is not a valid TruncationStrategy.
            Exception: Any other exception that may occur during preprocessing.
        """
        inputs = self._parse_and_tokenize(inputs, truncation=truncation, **kwargs)
        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        '''
        Forward pass method for the Text2TextGenerationPipeline class.

        Args:
            self: An instance of the Text2TextGenerationPipeline class.
            model_inputs (dict):
                A dictionary containing the model's input data.

                - input_ids (Tensor): A tensor representing the input sequence.
                  Shape: (batch_size, sequence_length)
                - Additional model-specific input tensors can be included.

        Returns:
            None: The method performs a forward pass on the model and updates the instance with the generated output.

        Raises:
            ValueError: If input_length is less than the specified min_length or greater than the specified max_length.
            Exception: Any other exception raised during the model generation process.
        '''
        in_b, input_length = model_inputs["input_ids"].shape

        self.check_inputs(
            input_length,
            generate_kwargs.get("min_length", self.model.config.min_length),
            generate_kwargs.get("max_length", self.model.config.max_length),
        )
        output_ids = self.model.generate(**model_inputs, **generate_kwargs)
        out_b = output_ids.shape[0]

        output_ids = output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])

        return {"output_ids": output_ids}

    def postprocess(self, model_outputs, return_type=ReturnType.TEXT, clean_up_tokenization_spaces=False):
        """
        Postprocesses the model outputs to generate the final records based on the specified return type.

        Args:
            self (Text2TextGenerationPipeline): The instance of the Text2TextGenerationPipeline class.
            model_outputs (dict): The model outputs containing the generated output_ids.
            return_type (ReturnType): The type of return value to be generated. Defaults to ReturnType.TEXT.
            clean_up_tokenization_spaces (bool): Flag indicating whether to clean up tokenization spaces. Defaults to False.

        Returns:
            list: A list of records containing the processed model outputs based on the specified return type.

        Raises:
            None.
        """
        records = []
        for output_ids in model_outputs["output_ids"][0]:
            if return_type == ReturnType.TENSORS:
                record = {f"{self.return_name}_token_ids": output_ids}
            elif return_type == ReturnType.TEXT:
                record = {
                    f"{self.return_name}_text": self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                }
            records.append(record)
        return records


class SummarizationPipeline(Text2TextGenerationPipeline):
    """
    Summarize news articles and other documents.

    This summarizing pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"summarization"`.

    The models that this pipeline can use are models that have been fine-tuned on a summarization task, which is
    currently, '*bart-large-cnn*', '*google-t5/t5-small*', '*google-t5/t5-base*', '*google-t5/t5-large*',
    '*google-t5/t5-3b*', '*google-t5/t5-11b*'. See the up-to-date
    list of available models on [hf-mirror.com/models](https://hf-mirror.com/models?filter=summarization). For a list
    of available parameters, see the [following
    documentation](https://hf-mirror.com/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

    Example:
        ```python
        >>> # use bart in pytorch
        >>> summarizer = pipeline("summarization")
        >>> summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)
        ...
        >>> # use t5 in tf
        >>> summarizer = pipeline("summarization", model="google-t5/t5-base", tokenizer="google-t5/t5-base", framework="tf")
        >>> summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)
        ```
    """
    # Used in the return key of the pipeline.
    return_name = "summary"

    def __call__(self, *args, **kwargs):
        r"""
        Summarize the text(s) given as inputs.

        Args:
            documents (*str* or `List[str]`):
                One or several articles (or one list of articles) to summarize.
            return_text (`bool`, *optional*, defaults to `True`):
                Whether or not to include the decoded texts in the outputs
            return_tensors (`bool`, *optional*, defaults to `False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).

        Returns:
            A list or a list of list of `dict`:
                Each result comes as a dictionary with the following keys:

                - **summary_text** (`str`, present when `return_text=True`) -- The summary of the corresponding input.
                - **summary_token_ids** (`torch.Tensor` or `tf.Tensor`, present when `return_tensors=True`) -- The token
                  ids of the summary.
        """
        result = super().__call__(*args, **kwargs)

        return result

    def check_inputs(self, input_length: int, min_length: int, max_length: int):
        """
        Checks whether there might be something wrong with given input with regard to the model.
        """
        if max_length < min_length:
            logger.warning(f"Your min_length={min_length} must be inferior than your max_length={max_length}.")

        if input_length < max_length:
            logger.warning(
                f"Your max_length is set to {max_length}, but your input_length is only {input_length}. Since this is "
                "a summarization task, where outputs shorter than the input are typically wanted, you might "
                f"consider decreasing max_length manually, e.g. summarizer('...', max_length={input_length // 2})"
            )


class TranslationPipeline(Text2TextGenerationPipeline):
    """
    Translates from one language to another.

    This translation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"translation_xx_to_yy"`.

    The models that this pipeline can use are models that have been fine-tuned on a translation task. See the
    up-to-date list of available models on [hf-mirror.com/models](https://hf-mirror.com/models?filter=translation).
    For a list of available parameters, see the [following
    documentation](https://hf-mirror.com/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

    Example:
        ```python
        >>> en_fr_translator = pipeline("translation_en_to_fr")
        >>> en_fr_translator("How old are you?")
        ```
    """
    # Used in the return key of the pipeline.
    return_name = "translation"

    def check_inputs(self, input_length: int, min_length: int, max_length: int):
        """
        Method to check the input length against specified minimum and maximum lengths in the TranslationPipeline class.

        Args:
            self: Reference to the current instance of the class.
            input_length (int): Length of the input to be checked.
            min_length (int): Minimum acceptable length for the input.
            max_length (int): Maximum acceptable length for the input.

        Returns:
            None.

        Raises:
            None: However, it may trigger a warning message if the input_length exceeds 90% of the max_length.
        """
        if input_length > 0.9 * max_length:
            logger.warning(
                f"Your input_length: {input_length} is bigger than 0.9 * max_length: {max_length}. You might consider "
                "increasing your max_length manually, e.g. translator('...', max_length=400)"
            )
        return True

    def preprocess(self, *args, truncation=TruncationStrategy.DO_NOT_TRUNCATE, src_lang=None, tgt_lang=None):
        """
        This method preprocesses the input for translation using the TranslationPipeline class.

        Args:
            self: The instance of the TranslationPipeline class.
                - Type: TranslationPipeline
                - Purpose: Represents the instance of the TranslationPipeline class.
            *args: Variable length argument list.
                - Type: tuple
                - Purpose: Represents the input arguments for the preprocessing.
            truncation: Specifies the truncation strategy for the input sequences.
                - Type: TruncationStrategy (Enum)
                - Purpose: Determines the truncation strategy for the input sequences.
                - Restrictions: Should be one of the TruncationStrategy enum values.
            src_lang: Specifies the source language for translation.
                - Type: str
                - Purpose: Represents the source language for translation.
            tgt_lang: Specifies the target language for translation.
                - Type: str
                - Purpose: Represents the target language for translation.

        Returns:
            None.

        Raises:
            None.
        """
        if getattr(self.tokenizer, "_build_translation_inputs", None):
            return self.tokenizer._build_translation_inputs(
                *args, return_tensors='ms', truncation=truncation, src_lang=src_lang, tgt_lang=tgt_lang
            )
        else:
            return super()._parse_and_tokenize(*args, truncation=truncation)

    def _sanitize_parameters(self, src_lang=None, tgt_lang=None, **kwargs):
        """
        Sanitizes the parameters for the TranslationPipeline.

        Args:
            self (TranslationPipeline): The instance of the TranslationPipeline class.
            src_lang (str, optional): The source language for translation. Defaults to None.
            tgt_lang (str, optional): The target language for translation. Defaults to None.

        Returns:
            tuple: A tuple containing the sanitized preprocess parameters, forward parameters, and postprocess parameters.

        Raises:
            None.

        Description:
            This method sanitizes the parameters for the TranslationPipeline by ensuring that the source language
            and target language are set correctly. It also handles the scenario where both the source language
            and target language are not provided, but the task name is given in the kwargs.
            In this case, the method extracts the source and target languages from the task name and sets them in the
            preprocess parameters.

            - If src_lang is provided, it updates the preprocess_params dictionary with the src_lang parameter.
            - If tgt_lang is provided, it updates the preprocess_params dictionary with the tgt_lang parameter.
            - If neither src_lang nor tgt_lang is provided, it checks if the task name is present in kwargs.
            - If it is, and the task name is in the format 'prefix_srcLang_to_tgtLang_suffix', it extracts the source
            language and target language from the task name and sets them in the preprocess_params dictionary.

            The method then returns the sanitized preprocess parameters, forward parameters, and postprocess parameters as a tuple.
        """
        preprocess_params, forward_params, postprocess_params = super()._sanitize_parameters(**kwargs)
        if src_lang is not None:
            preprocess_params["src_lang"] = src_lang
        if tgt_lang is not None:
            preprocess_params["tgt_lang"] = tgt_lang
        if src_lang is None and tgt_lang is None:
            # Backward compatibility, direct arguments use is preferred.
            task = kwargs.get("task", self.task)
            items = task.split("_")
            if task and len(items) == 4:
                # translation, XX, to YY
                preprocess_params["src_lang"] = items[1]
                preprocess_params["tgt_lang"] = items[3]
        return preprocess_params, forward_params, postprocess_params

    def __call__(self, *args, **kwargs):
        r"""
        Translate the text(s) given as inputs.

        Args:
            args (`str` or `List[str]`):
                Texts to be translated.
            return_tensors (`bool`, *optional*, defaults to `False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (`bool`, *optional*, defaults to `True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the potential extra spaces in the text output.
            src_lang (`str`, *optional*):
                The language of the input. Might be required for multilingual models. Will not have any effect for
                single pair translation models
            tgt_lang (`str`, *optional*):
                The language of the desired output. Might be required for multilingual models. Will not have any effect
                for single pair translation models
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).

        Returns:
            A list or a list of list of `dict`:
                Each result comes as a dictionary with the following keys:

                - **translation_text** (`str`, present when `return_text=True`) -- The translation.
                - **translation_token_ids** (`torch.Tensor` or `tf.Tensor`, present when `return_tensors=True`) -- The
                  token ids of the translation.
        """
        result = super().__call__(*args, **kwargs)
        return result
