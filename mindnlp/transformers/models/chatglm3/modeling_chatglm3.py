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
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=consider-using-f-string
# pylint: disable=arguments-renamed
""" MindSpore ChatGLM3 model. """

import copy
import warnings
from typing import Optional, List, Callable, Dict
import mindspore
from mindspore import ops

from mindnlp.utils import logging
from mindnlp.transformers.models.chatglm2.modeling_chatglm2 import InvalidScoreLogitsProcessor, \
    ChatGLM2Model, ChatGLM2ForConditionalGeneration, ChatGLM2ForSequenceClassification
from ...generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig

logger = logging.get_logger(__name__)


CHATGLM3_6B_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "THUDM/chatglm3-6b",
    # See all ChatGLM models at https://huggingface.co/models?filter=chatglm
]


class ChatGLM3Model(ChatGLM2Model):
    """ChatGLM3Model"""

class ChatGLM3ForConditionalGeneration(ChatGLM2ForConditionalGeneration):
    """ChatGLM3ForConditionalGeneration"""
    def process_response(self, output, history):
        content = ""
        history = copy.deepcopy(history)
        for response in output.split("<|assistant|>"):
            if "\n" in response:
                metadata, content = response.split("\n", maxsplit=1)
            else:
                metadata, content = "", response
            if not metadata.strip():
                content = content.strip()
                history.append({"role": "assistant", "metadata": metadata, "content": content})
                content = content.replace("[[训练时间]]", "2023年")
            else:
                history.append({"role": "assistant", "metadata": metadata, "content": content})
                if history[0]["role"] == "system" and "tools" in history[0]:
                    content = "\n".join(content.split("\n")[1:-1])
                    parameters = eval(content) # pylint: disable=eval-used
                    content = {"name": metadata.strip(), "parameters": parameters}
                else:
                    content = {"name": metadata.strip(), "content": content}
        return content, history

    def chat(self, tokenizer, query: str, history: List[Dict] = None, role: str = "user",
             max_length: int = 8192, num_beams=1, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
             **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        inputs = tokenizer.build_chat_input(query, history=history, role=role)
        inputs = inputs.to(self.device)
        eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                        tokenizer.get_command("<|observation|>")]
        outputs = self.generate(**inputs, **gen_kwargs, eos_token_id=eos_token_id)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
        response = tokenizer.decode(outputs)
        history.append({"role": role, "content": query})
        response, history = self.process_response(response, history)
        return response, history

    def stream_chat(self, tokenizer, query: str, history: List[Dict] = None, role: str = "user",
                    past_key_values=None,max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8,
                    logits_processor=None, return_past_key_values=False, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                        tokenizer.get_command("<|observation|>")]
        gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if past_key_values is None:
            inputs = tokenizer.build_chat_input(query, history=history, role=role)
        else:
            inputs = tokenizer.build_chat_input(query, role=role)

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[0]
            if self.transformer.pre_seq_len is not None:
                past_length -= self.transformer.pre_seq_len
            inputs['position_ids'] = inputs.position_ids + past_length
            attention_mask = inputs.attention_mask
            attention_mask = ops.cat((attention_mask.new_ones((1, past_length)), attention_mask), axis=1)
            inputs['attention_mask'] = attention_mask
        history.append({"role": role, "content": query})
        for outputs in self.stream_generate(**inputs, past_key_values=past_key_values,
                                            eos_token_id=eos_token_id, return_past_key_values=return_past_key_values,
                                            **gen_kwargs):
            if return_past_key_values:
                outputs, past_key_values = outputs
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
            response = tokenizer.decode(outputs)
            if response and response[-1] != "�":
                response, new_history = self.process_response(response, history)
                if return_past_key_values:
                    yield response, new_history, past_key_values
                else:
                    yield response, new_history

    def stream_generate(
            self,
            input_ids,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, mindspore.Tensor], List[int]]] = None,
            return_past_key_values=False,
            **kwargs,
    ):
        _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        model_kwargs["use_cache"] = generation_config.use_cache
        _, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        eos_token_id_tensor = mindspore.tensor(eos_token_id) if eos_token_id is not None else None

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            if not has_default_max_length:
                logger.warn(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        logits_warper = self._get_logits_warper(generation_config)

        unfinished_sequences = ops.ones(input_ids.shape[0], dtype=input_ids.dtype)
        scores = None
        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = ops.softmax(next_token_scores, axis=-1)
            if generation_config.do_sample:
                next_tokens = ops.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = ops.argmax(probs, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = ops.cat([input_ids, next_tokens[:, None]], axis=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile((eos_token_id_tensor.shape[0], 1)).ne(eos_token_id_tensor.unsqueeze(1)).prod(axis=0)
            )

            if return_past_key_values:
                yield input_ids, outputs.past_key_values
            else:
                yield input_ids

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

class ChatGLM3ForSequenceClassification(ChatGLM2ForSequenceClassification):
    """ChatGLM3ForSequenceClassification"""

__all__ = [
    'CHATGLM3_6B_PRETRAINED_MODEL_ARCHIVE_LIST',
    'ChatGLM3Model',
    'ChatGLM3ForSequenceClassification',
    'ChatGLM3ForConditionalGeneration'
]
