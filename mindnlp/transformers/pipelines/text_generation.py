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
"""text generation pipeline."""

import enum
import warnings
from typing import Dict,List
from .base import Pipeline

class ReturnType(enum.Enum):

    """
    Represents the possible return types for a function.
    
    This class inherits from enum.Enum and defines various return types that a function can have. 
    The return types can be used to specify the expected type of value that a function should return.
    
    Attributes:
        SUCCESS: Represents a successful return from a function.
        FAILURE: Represents a failure return from a function.
        ERROR: Represents an error return from a function.
        NONE: Represents a return with no value.
    """
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2


class Chat:
    """
    This class is intended to just be used internally in this pipeline and not exposed to users. We convert chats
    to this format because the rest of the pipeline code tends to assume that lists of messages are
    actually a batch of samples rather than messages in the same conversation."""
    def __init__(self, messages: List[Dict[str, str]]):
        """
        Initializes a new instance of the Chat class.
        
        Args:
            self: Represents the instance of the class.
            messages (List[Dict[str, str]]): A list of dictionaries representing chat messages.
                Each dictionary must contain 'role' and 'content' keys.
            
        Returns:
            None.
        
        Raises:
            ValueError: Raised if any dictionary in the messages list does not contain both 'role' and 'content' keys.
        """
        for message in messages:
            if not ("role" in message and "content" in message):
                raise ValueError("When passing chat dicts as input, each dict must have a 'role' and 'content' key.")
        self.messages = messages


class TextGenerationPipeline(Pipeline):
    """
    Language generation pipeline using any `ModelWithLMHead`. This pipeline predicts the words that will follow a
    specified text prompt. It can also accept one or more chats. Each chat takes the form of a list of dicts,
    where each dict contains "role" and "content" keys.

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial). You can pass text
    generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about
    text generation parameters in [Text generation strategies](../generation_strategies) and [Text
    generation](text_generation).

    This language generation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"text-generation"`.

    The models that this pipeline can use are models that have been trained with an autoregressive language modeling
    objective, which includes the uni-directional models in the library (e.g. openai-community/gpt2). See the list of available models
    on [hf-mirror.com/models](https://hf-mirror.com/models?filter=text-generation).
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes an instance of the TextGenerationPipeline class.
        
        Args:
            self: The instance of the TextGenerationPipeline class.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__(*args, **kwargs)

        if "prefix" not in self._preprocess_params:
            prefix = None
            if self.model.config.prefix is not None:
                prefix = self.model.config.prefix
            if prefix is not None:
                preprocess_params, forward_params, _ = self._sanitize_parameters(prefix=prefix, **self._forward_params)
                self._preprocess_params = {**self._preprocess_params, **preprocess_params}
                self._forward_params = {**self._forward_params, **forward_params}

    def _sanitize_parameters(
        self,
        return_full_text=None,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        prefix=None,
        handle_long_generation=None,
        stop_sequence=None,
        add_special_tokens=False,
        truncation=None,
        padding=False,
        max_length=None,
        **generate_kwargs,
    ):
        """
        This method '_sanitize_parameters' in the class 'TextGenerationPipeline' is responsible for sanitizing and
        processing parameters used in text generation.
        It takes 13 parameters: self, return_full_text, return_tensors, return_text, return_type,
        clean_up_tokenization_spaces, prefix, handle_long_generation, stop_sequence, add_special_tokens,
        truncation, padding, max_length.

        Args:
            self: The instance of the class.
            return_full_text (bool): Whether to return the full generated text.
            return_tensors (bool): Whether to return the generated text as tensors.
            return_text (bool): Whether to return the generated text as plain text.
            return_type (ReturnType): The type of text to return.
            clean_up_tokenization_spaces (bool): Whether to clean up tokenization spaces.
            prefix (str): The prefix to be added to the input text.
            handle_long_generation (str): The strategy to handle long text generation, expected values are [None, 'hole'].
            stop_sequence (str): The sequence to stop text generation at.
            add_special_tokens (bool): Whether to add special tokens during text generation.
            truncation (bool): Whether to truncate the generated text.
            padding (bool): Whether to pad the generated text.
            max_length (int): The maximum length of the generated text.

        Returns:
            None.

        Raises:
            ValueError: If the provided 'handle_long_generation' is not a valid value.
            ValueError: If 'return_text' is provided while 'return_full_text' is also specified.
            ValueError: If 'return_tensors' is provided while 'return_full_text' is also specified.
            ValueError: If 'return_text' is provided while 'return_tensors' is also specified.
            ValueError: If 'handle_long_generation' is not one of the expected values [None, 'hole'].
            Warning: If stopping on a multiple token sequence is attempted, as it is not yet supported.
        """
        preprocess_params = {
            "add_special_tokens": add_special_tokens,
            "truncation": truncation,
            "padding": padding,
            "max_length": max_length,
        }
        if max_length is not None:
            generate_kwargs["max_length"] = max_length

        if prefix is not None:
            preprocess_params["prefix"] = prefix
        if prefix:
            prefix_inputs = self.tokenizer(
                prefix, padding=False, add_special_tokens=add_special_tokens, return_tensors='ms'
            )
            generate_kwargs["prefix_length"] = prefix_inputs["input_ids"].shape[-1]

        if handle_long_generation is not None:
            if handle_long_generation not in {"hole"}:
                raise ValueError(
                    f"{handle_long_generation} is not a valid value for `handle_long_generation` parameter expected"
                    " [None, 'hole']"
                )
            preprocess_params["handle_long_generation"] = handle_long_generation

        preprocess_params.update(generate_kwargs)
        forward_params = generate_kwargs

        postprocess_params = {}
        if return_full_text is not None and return_type is None:
            if return_text is not None:
                raise ValueError("`return_text` is mutually exclusive with `return_full_text`")
            if return_tensors is not None:
                raise ValueError("`return_full_text` is mutually exclusive with `return_tensors`")
            return_type = ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
        if return_tensors is not None and return_type is None:
            if return_text is not None:
                raise ValueError("`return_text` is mutually exclusive with `return_tensors`")
            return_type = ReturnType.TENSORS
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        if stop_sequence is not None:
            stop_sequence_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
            if len(stop_sequence_ids) > 1:
                warnings.warn(
                    "Stopping on a multiple token sequence is not yet supported on transformers. The first token of"
                    " the stop sequence will be used as the stop sequence string in the interim."
                )
            generate_kwargs["eos_token_id"] = stop_sequence_ids[0]

        return preprocess_params, forward_params, postprocess_params

    def __call__(self, text_inputs, **kwargs):
        """
        Complete the prompt(s) given as inputs.

        Args:
            text_inputs (`str` or `List[str]`):
                One or several prompts (or one list of prompts) to complete.
            return_tensors (`bool`, *optional*, defaults to `False`):
                Whether or not to return the tensors of predictions (as token indices) in the outputs. If set to
                `True`, the decoded text is not returned.
            return_text (`bool`, *optional*, defaults to `True`):
                Whether or not to return the decoded texts in the outputs.
            return_full_text (`bool`, *optional*, defaults to `True`):
                If set to `False` only added text is returned, otherwise the full text is returned. Only meaningful if
                *return_text* is set to True.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
                Whether or not to clean up the potential extra spaces in the text output.
            prefix (`str`, *optional*):
                Prefix added to prompt.
            handle_long_generation (`str`, *optional*):
                By default, this pipelines does not handle long generation (ones that exceed in one form or the other
                the model maximum length). There is no perfect way to adress this (more info
                :https://github.com/huggingface/transformers/issues/14033#issuecomment-948385227). This provides common
                strategies to work around that problem depending on your use case.

                - `None` : default strategy where nothing in particular happens
                - `"hole"`: Truncates left of input, and leaves a gap wide enough to let generation happen (might
                  truncate a lot of the prompt and not suitable when generation exceed the model capacity)
            generate_kwargs (`dict`, *optional*):
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).

        Returns:
            A list or a list of list of `dict`:
                Returns one of the following dictionaries (cannot return a combination of both `generated_text` and
                `generated_token_ids`):

                - **generated_text** (`str`, present when `return_text=True`) -- The generated text.
                - **generated_token_ids** (`torch.Tensor` or `tf.Tensor`, present when `return_tensors=True`) -- The token
                  ids of the generated text.
        """
        if isinstance(text_inputs, (list, tuple)) and isinstance(text_inputs[0], (list, tuple, dict)):
            # We have one or more prompts in list-of-dicts format, so this is chat mode
            if isinstance(text_inputs[0], dict):
                return super().__call__(Chat(text_inputs), **kwargs)
            else:
                chats = [Chat(chat) for chat in text_inputs]  # 🐈 🐈 🐈
                return super().__call__(chats, **kwargs)
        else:
            return super().__call__(text_inputs, **kwargs)

    def preprocess(
        self,
        prompt_text,
        prefix="",
        handle_long_generation=None,
        add_special_tokens=False,
        truncation=None,
        padding=False,
        max_length=None,
        **generate_kwargs,
    ):
        """
        Preprocesses the prompt text for text generation.

        Args:
            self: An instance of the TextGenerationPipeline class.
            prompt_text (Union[str, Chat]): The text or chat prompt to preprocess.
            prefix (str, optional): A prefix to add to the prompt text. Default is an empty string.
            handle_long_generation (str, optional): Specifies how to handle long generation. Default is None.
            add_special_tokens (bool, optional): Whether to add special tokens to the input text. Default is False.
            truncation (bool or str, optional): Specifies whether or how to truncate the input text. Default is None.
            padding (bool, optional): Whether to pad the input text. Default is False.
            max_length (int, optional): The maximum length of the input text. Default is None.
            **generate_kwargs: Additional keyword arguments to be passed to the text generation process.

        Returns:
            dict: A dictionary containing the preprocessed inputs for text generation.
                The dictionary includes the following keys:

                - 'input_ids' (torch.Tensor): The tokenized input text.
                - 'attention_mask' (torch.Tensor, optional): The attention mask for the input text, if padding is enabled.
                - 'prompt_text' (Union[str, Chat]): The original prompt text.

        Raises:
            ValueError: If the number of new tokens exceeds the model's maximum length.
            ValueError: If the number of desired tokens exceeds the model's maximum length and 'hole' handling is used.

        """
        if isinstance(prompt_text, Chat):
            inputs = self.tokenizer.apply_chat_template(
                prompt_text.messages,
                truncation=truncation,
                padding=padding,
                max_length=max_length,
                add_generation_prompt=True,
                return_tensors='ms',
                return_dict= True
            )
        else:
            inputs = self.tokenizer(
                prefix + prompt_text,
                truncation=truncation,
                padding=padding,
                max_length=max_length,
                add_special_tokens=add_special_tokens,
                return_tensors='ms',
            )
        inputs["prompt_text"] = prompt_text

        if handle_long_generation == "hole":
            cur_len = inputs["input_ids"].shape[-1]
            if "max_new_tokens" in generate_kwargs:
                new_tokens = generate_kwargs["max_new_tokens"]
            else:
                new_tokens = generate_kwargs.get("max_length", self.model.config.max_length) - cur_len
                if new_tokens < 0:
                    raise ValueError("We cannot infer how many new tokens are expected")
            if cur_len + new_tokens > self.tokenizer.model_max_length:
                keep_length = self.tokenizer.model_max_length - new_tokens
                if keep_length <= 0:
                    raise ValueError(
                        "We cannot use `hole` to handle this generation the number of desired tokens exceeds the"
                        " models max length"
                    )

                inputs["input_ids"] = inputs["input_ids"][:, -keep_length:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -keep_length:]

        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        """
        This method, '_forward', is part of the 'TextGenerationPipeline' class and is responsible for
        generating text based on the provided model inputs and generation parameters.

        Args:
            self (object): The instance of the TextGenerationPipeline class.
            model_inputs (dict):
                A dictionary containing the model inputs required for text generation.

                - input_ids (Tensor): The input token IDs for the model.
                - attention_mask (Tensor, optional): The attention mask for the model inputs. Defaults to None.
                - prompt_text (str): The prompt text to influence the text generation.

            **generate_kwargs (dict): Additional keyword arguments for text generation, such as 'max_length',
            'min_length', 'prefix_length', etc.

        Returns:
            dict:
                A dictionary containing the generated text sequence, the input token IDs, and the prompt text.

                - generated_sequence (Tensor): The generated text sequence.
                - input_ids (Tensor): The input token IDs used for generation.
                - prompt_text (str): The prompt text used for generation.

        Raises:
            ValueError: If the input_ids shape is invalid (e.g., input_ids.shape[1] == 0).
            KeyError: If 'prompt_text' or 'prefix_length' is missing from the model_inputs or generate_kwargs respectively.
            RuntimeError: If an error occurs during the text generation process.
        """
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]
        prompt_text = model_inputs.pop("prompt_text")

        # If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
        # generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
        prefix_length = generate_kwargs.pop("prefix_length", 0)
        if prefix_length > 0:
            has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].max_new_tokens is not None
            )
            if not has_max_new_tokens:
                generate_kwargs["max_length"] = generate_kwargs.get("max_length") or self.model.config.max_length
                generate_kwargs["max_length"] += prefix_length
            has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].min_new_tokens is not None
            )
            if not has_min_new_tokens and "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length

        # BS x SL
        generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        out_b = generated_sequence.shape[0]
        generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text}

    def postprocess(self, model_outputs, return_type=ReturnType.FULL_TEXT, clean_up_tokenization_spaces=True):
        """
        postprocess method in the TextGenerationPipeline class.

        Args:
            self: The instance of the TextGenerationPipeline class.
            model_outputs (dict): A dictionary containing model outputs including 'generated_sequence', 'input_ids',
                and 'prompt_text'.
            return_type (ReturnType): An enum specifying the type of return value desired.
                Can be one of the following: ReturnType.TENSORS, ReturnType.NEW_TEXT, or ReturnType.FULL_TEXT.
            clean_up_tokenization_spaces (bool): A flag indicating whether to clean up tokenization spaces in
                the generated text.

        Returns:
            list: A list of dictionaries containing the post-processed output based on the specified return_type.
                Each dictionary in the list may have the following keys based on the return_type:

                - 'generated_token_ids': List of token ids if return_type is ReturnType.TENSORS.
                - 'generated_text': The generated text if return_type is ReturnType.NEW_TEXT or ReturnType.FULL_TEXT.
        
        Raises:
            TypeError: If model_outputs is not a dictionary or return_type is not a valid ReturnType enum.
            ValueError: If return_type is not one of the valid enum values.
        """
        generated_sequence = model_outputs["generated_sequence"][0]
        input_ids = model_outputs["input_ids"]
        prompt_text = model_outputs["prompt_text"]
        generated_sequence = generated_sequence.numpy().tolist()
        records = []
        for sequence in generated_sequence:
            if return_type == ReturnType.TENSORS:
                record = {"generated_token_ids": sequence}
            elif return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
                # Decode text
                text = self.tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )

                # Remove PADDING prompt of the sequence if XLNet or Transfo-XL model is used
                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        self.tokenizer.decode(
                            input_ids[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        )
                    )

                all_text = text[prompt_length:]
                if return_type == ReturnType.FULL_TEXT:
                    if isinstance(prompt_text, str):
                        all_text = prompt_text + all_text
                    elif isinstance(prompt_text, Chat):
                        all_text = prompt_text.messages + [{"role": "assistant", "content": all_text}]

                record = {"generated_text": all_text}
            records.append(record)

        return records
