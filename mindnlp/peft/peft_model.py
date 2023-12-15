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
"""PEFT model."""
# pylint: disable=E0602  # Import outside toplevel
# pylint: disable=R1705  # Unnecessary "else" after "return"
# pylint: disable=W0613  # unused kwargs
# pylint: disable=C0415  # Undefined variable
import os
import warnings
import inspect
from contextlib import contextmanager
from copy import deepcopy
from typing import Dict

from mindspore import nn, ops
from mindspore.train.serialization import  _exec_save
from mindspore.nn import CrossEntropyLoss

from .config import PeftConfig, PromptLearningConfig

from .tuners import (
    LoraModel,
    # LoraConfig
)
from .utils import (
    # SAFETENSORS_WEIGHTS_NAME,
    # TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    WEIGHTS_NAME,
    PeftType,
    # TaskType,
    # _get_batch_size,
    _prepare_prompt_learning_config,
    # _set_adapter,
    _set_trainable,
    # add_library_to_model_card,
    get_peft_model_state_dict,
    # infer_device,
    load_peft_weights,
    set_peft_model_state_dict,
    shift_tokens_right,
    # _get_batch_size, # will be used for prompt learning methods
)


PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.LORA: LoraModel,
}
class PeftModel(nn.Cell):
    """
    Base model encompassing various Peft methods.

    Args:
        model ([`~mindnlp.models.PreTrainedModel`]): The base transformer model used for Peft.
        peft_config ([`PeftConfig`]): The configuration of the Peft model.
    """

    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        super().__init__()
        self.base_model = model
        self.config = getattr(self.base_model, "config", {"model_type": "custom"})
        self.modules_to_save = None
        self.peft_config: Dict[str, PeftConfig] = {}
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        # self.base_model_torch_dtype = getattr(model, "dtype", None)
        if not peft_config.is_prompt_learning:
            self.peft_config[adapter_name] = peft_config
            self.base_model = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type](
                self.base_model, self.peft_config, adapter_name
            )
            self.set_additional_trainable_modules(peft_config, adapter_name)
        else:
            self.add_adapter(adapter_name, peft_config)

        # if getattr(model, "is_gradient_checkpointing", True):
        #     model = self._prepare_model_for_gradient_checkpointing(model)
        # if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
        #     self.base_model.config.pretraining_tp = 1



    def save_pretrained(self, save_directory, **kwargs):
        r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
        method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)

        for adapter_name, peft_config in self.peft_config.items():
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                self,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name
            )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            _exec_save(
                ckpt_file_name=os.path.join(output_dir, WEIGHTS_NAME),
                data_list=output_state_dict,
            )

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None),
                    self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True
            peft_config.save_pretrained(output_dir)
            peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(cls, model, model_id, adapter_name="default", is_trainable=False, **kwargs):
        r"""
        Instantiate a [`LoraModel`] from a pretrained Lora configuration and weights.

        Args:
            model ([`~transformers.PreTrainedModel`]):
                The model to be adapted. The model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`] method from the 🤗 Transformers library.
            model_id (`str` or `os.PathLike`):
                The name of the Lora configuration to use. Can be :
                    - A path to a directory containing a Lora configuration file saved using the `save_pretrained`
                      method (`./my_lora_config_directory/`).
        """
        from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING  # pylint: disable=R0401

        # load peft config
        config = PEFT_TYPE_TO_CONFIG_MAPPING[
            PeftConfig.from_pretrained(model_id, subfolder=kwargs.get("subfolder", None)).peft_type
        ].from_pretrained(model_id, subfolder=kwargs.get("subfolder", None))

        config.inference_mode = not is_trainable

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING:
            model = cls(model, config, adapter_name)
        else:
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config, adapter_name)
        model.load_adapter(model_id, adapter_name, **kwargs)
        return model


    def load_adapter(self, model_id: str, adapter_name: str, is_trainable: bool = False, **kwargs):
        """load adapter to peft model, called by `model.from_pretrained`."""
        # NOTE: remove download logic.
        if adapter_name not in self.peft_config:
            raise ValueError(f"{adapter_name} is not a valid adapter name. Valid names: {self.peft_config.keys()}")

        adapters_weights = load_peft_weights(model_id)

        # load the weights into the model
        load_result = set_peft_model_state_dict(self, adapters_weights, adapter_name=adapter_name)
        # TODO: add parallel logic & offload logic & device map logic(dispatch_model)

        # Set model in evaluation mode to deactivate Dropout modules by default
        if not is_trainable:
            self.set_train(False)

        return load_result


    def get_nb_trainable_parameters(self):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for param in self.get_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param


    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)

    def construct(self, *args, **kwargs):
        """
        Forward pass of the model.
        """
        # print(self.get_base_model().layers[0].__class__.construct)
        return self.get_base_model()(*args, **kwargs)

    @contextmanager
    def disable_adapter(self):
        """
        Disables the adapter module.
        """
        try:
            self.base_model.disable_adapter_layers()
            yield
        finally:
            self.base_model.enable_adapter_layers()

    def get_base_model(self):
        """
        Returns the base model.
        """
        return self.base_model.model
        # return self.base_model if self.active_peft_config.is_prompt_learning else self.base_model.model


    def add_adapter(self, adapter_name: str, peft_config: PeftConfig):
        """add adapter."""
        if peft_config.peft_type != self.peft_type:
            raise ValueError(
                f"Cannot combine adapters with different peft types. "
                f"Found {self.peft_type} and {peft_config.peft_type}."
            )

        self.peft_config[adapter_name] = peft_config

        try:
            if peft_config.is_prompt_learning:  # add_adapter methods for prompt learning setup
                if hasattr(self.config, "to_dict"):
                    dict_config = self.config.to_dict()
                else:
                    dict_config = self.config

                peft_config = _prepare_prompt_learning_config(peft_config, dict_config)
                self._setup_prompt_encoder(adapter_name)
            # elif peft_config.is_adaption_prompt:
            #     self.base_model.add_adapter(adapter_name, peft_config)
            else:
                # inject adapter into base model (load model instead of initialize new one)
                self.base_model.inject_adapter(self, adapter_name)
        except Exception:  # somthing went wrong, roll back
            del self.peft_config[adapter_name]
            raise

        self.set_additional_trainable_modules(peft_config, adapter_name)


    def set_additional_trainable_modules(self, peft_config, adapter_name):
        """set additional trainable cells"""
        if getattr(peft_config, "modules_to_save", None) is not None:
            if self.modules_to_save is None:
                self.modules_to_save = set(peft_config.modules_to_save)
            else:
                self.modules_to_save.update(peft_config.modules_to_save)
            _set_trainable(self, adapter_name)

    @property
    def active_peft_config(self):
        """active_peft_config"""
        return self.peft_config[self.active_adapter]


class PeftModelForSequenceClassification(PeftModel):
    """
    Peft model for sequence classification tasks.

    Args:
        model ([`~mindnlp.models.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.

    """

    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        if self.modules_to_save is None:
            self.modules_to_save = {"classifier", "score"}
        else:
            self.modules_to_save.update({"classifier", "score"})

        for name, _ in self.base_model.cells_and_names():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self, adapter_name)

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            # NOTE:some args not exists in base model
            # inputs_embeds=inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        raise NotImplementedError
        # batch_size = _get_batch_size(input_ids, inputs_embeds)
        # if attention_mask is not None:
        #     # concat prompt attention mask
        #     prefix_attention_mask = ops.ones(batch_size, peft_config.num_virtual_tokens)
        #     attention_mask = ops.cat((prefix_attention_mask, attention_mask), axis=1)
        # if kwargs.get("position_ids", None) is not None:
        #     warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
        #     kwargs["position_ids"] = None
        # kwargs.update(
        #     {
        #         "attention_mask": attention_mask,
        #         "labels": labels,
        #         "output_attentions": output_attentions,
        #         "output_hidden_states": output_hidden_states,
        #         "return_dict": return_dict,
        #     }
        # )

        # if peft_config.peft_type == PeftType.PREFIX_TUNING:
        #     return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        # if kwargs.get("token_type_ids", None) is not None:
        #     kwargs["token_type_ids"] = ops.cat(
        #         (
        #             ops.zeros(batch_size, peft_config.num_virtual_tokens),
        #             kwargs["token_type_ids"],
        #         ),
        #         axis=1,
        #     ).long()
        # if inputs_embeds is None:
        #     inputs_embeds = self.word_embeddings(input_ids)
        # prompts = self.get_prompt(batch_size=batch_size)
        # prompts = prompts.to(inputs_embeds.dtype)
        # inputs_embeds = ops.cat((prompts, inputs_embeds), axis=1)
        # return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

class PeftModelForCausalLM(PeftModel):
    """
    Peft model for causal language modeling.

    Args:
        model ([`~mindnlp.models.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.
    """

    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not isinstance(peft_config, PromptLearningConfig):
            if self.base_model.config.model_type == "mpt":
                if inputs_embeds is not None:
                    raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = ops.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = ops.cat((prefix_attention_mask, attention_mask), axis=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # concat prompt labels
        if labels is not None:
            prefix_labels = ops.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
            kwargs["labels"] = ops.cat((prefix_labels, labels), axis=1)
        prompts = self.get_prompt(batch_size=batch_size)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = ops.cat((prompts, inputs_embeds), axis=1)
        return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def generate(self, **kwargs):
        """generate."""
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise

        self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """prepare_inputs_for_generation."""
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if isinstance(peft_config, PromptLearningConfig):
            if model_kwargs.get("attention_mask", None) is not None:
                prefix_attention_mask = ops.ones(
                    model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens)
                model_kwargs["attention_mask"] = ops.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), axis=1
                )

            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                model_kwargs["position_ids"] = None

            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                model_kwargs["past_key_values"] = past_key_values
            else:
                if model_kwargs["past_key_values"] is None:
                    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                    prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                    prompts = prompts.to(inputs_embeds.dtype)
                    model_kwargs["inputs_embeds"] = ops.cat((prompts, inputs_embeds), axis=1)
                    model_kwargs["input_ids"] = None

        return model_kwargs

class PeftModelForSeq2SeqLM(PeftModel):
    """
    Peft model for sequence-to-sequence language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.

    """

    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not isinstance(peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if decoder_attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = ops.ones(batch_size, peft_config.num_virtual_tokens).to(
                decoder_attention_mask.device
            )
            decoder_attention_mask = ops.cat((prefix_attention_mask, decoder_attention_mask), axis=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(
                input_ids=input_ids, decoder_input_ids=decoder_input_ids, past_key_values=past_key_values, **kwargs
            )
        elif peft_config.peft_type in [PeftType.PROMPT_TUNING, PeftType.P_TUNING]:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)

            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = ops.ones(batch_size, peft_config.num_virtual_tokens).to(
                    attention_mask.device
                )
                kwargs["attention_mask"] = ops.cat((prefix_attention_mask, attention_mask), axis=1)

            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = ops.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), axis=1)

            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            if decoder_inputs_embeds is None and decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                decoder_inputs_embeds = self.word_embeddings(decoder_input_ids)

            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = ops.ones(batch_size, peft_config.num_virtual_tokens).to(
                    attention_mask.device
                )
                kwargs["attention_mask"] = ops.cat((prefix_attention_mask, attention_mask), axis=1)
            # concat prompt labels
            if labels is not None:
                if peft_config.num_transformer_submodules == 1:
                    kwargs["labels"] = labels
                elif peft_config.num_transformer_submodules == 2:
                    prefix_labels = ops.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                    kwargs["labels"] = ops.cat((prefix_labels, labels), axis=1)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = ops.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), axis=1)
            if peft_config.num_transformer_submodules == 1:
                return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
            elif peft_config.num_transformer_submodules == 2:
                decoder_inputs_embeds = ops.cat(
                    (prompts[:, peft_config.num_virtual_tokens :], decoder_inputs_embeds), axis=1
                )
                return self.base_model(
                    inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, **kwargs
                )
            return None # never go here

    def generate(self, **kwargs):
        """generate."""
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self._prepare_encoder_decoder_kwargs_for_generation
        )
        try:
            if not isinstance(peft_config, PromptLearningConfig):
                outputs = self.base_model.generate(**kwargs)
            else:
                if "input_ids" not in kwargs:
                    raise ValueError("input_ids must be provided for Peft model generation")
                if kwargs.get("position_ids", None) is not None:
                    warnings.warn(
                        "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
                    )
                    kwargs["position_ids"] = None
                if kwargs.get("token_type_ids", None) is not None:
                    warnings.warn(
                        "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                    )
                    kwargs["token_type_ids"] = None

                if peft_config.peft_type == PeftType.PREFIX_TUNING:
                    outputs = self.base_model.generate(**kwargs)
                elif peft_config.peft_type in [PeftType.PROMPT_TUNING, PeftType.P_TUNING]:
                    kwargs = deepcopy(kwargs)

                    if "encoder_outputs" in kwargs:
                        del kwargs["encoder_ouputs"]
                        warnings.warn(
                            "`encoder_outputs` should not be passed to `generate` when using prompt tuning. Ignoring it."
                        )

                    input_ids = kwargs.pop("input_ids")
                    inputs_embeds = self.word_embeddings(input_ids)
                    batch_size = inputs_embeds.shape[0]
                    prompts = self.get_prompt(batch_size=batch_size)
                    prompts = prompts.to(inputs_embeds.dtype)

                    inputs_embeds = ops.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), axis=1)
                    kwargs["inputs_embeds"] = inputs_embeds

                    if "attention_mask" in kwargs:
                        prefix_attention_mask = ops.ones(batch_size, peft_config.num_virtual_tokens).to(
                            kwargs["attention_mask"].device
                        )
                        kwargs["attention_mask"] = ops.cat((prefix_attention_mask, kwargs["attention_mask"]), axis=1)

                    return self.base_model.generate(**kwargs)
                else:
                    raise NotImplementedError
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            raise
        self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model_prepare_encoder_decoder_kwargs_for_generation
        )
        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """prepare inputs for generation"""
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
            batch_size = model_kwargs["decoder_input_ids"].shape[0]
            past_key_values = self.get_prompt(batch_size)
            model_kwargs["past_key_values"] = past_key_values

        return model_kwargs

class PeftModelForTokenClassification(PeftModel):
    """
    Peft model for token classification tasks.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.
    """

    def __init__(self, model, peft_config: PeftConfig = None, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        if self.modules_to_save is None:
            self.modules_to_save = {"classifier", "score"}
        else:
            self.modules_to_save.update({"classifier", "score"})

        for name, _ in self.base_model.cells_and_names():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self, adapter_name)

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not isinstance(peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = ops.ones(batch_size, peft_config.num_virtual_tokens)
            attention_mask = ops.cat((prefix_attention_mask, attention_mask), axis=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = ops.cat(
                    (
                        ops.zeros(batch_size, peft_config.num_virtual_tokens).to(self.word_embeddings.weight.device),
                        kwargs["token_type_ids"],
                    ),
                    axis=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = ops.cat((prompts, inputs_embeds), axis=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        kwargs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
            }
        )
        if "past_key_values" in fwd_params:
            return self.base_model(labels=labels, **kwargs)
        else:
            transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            sequence_output = outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.cells_and_names())]:
                sequence_output = self.base_model.dropout(sequence_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(sequence_output)

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
