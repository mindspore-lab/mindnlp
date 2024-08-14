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
import os
import warnings
import inspect
from contextlib import contextmanager
from copy import deepcopy
from typing import Dict, Optional

import mindspore
from mindspore import Tensor
from mindspore.train.serialization import _exec_save

from mindnlp.core import nn, ops
from .config import PeftConfig, PromptLearningConfig
from ..transformers import PreTrainedModel

from .tuners import (
    AdaLoraModel,
    AdaptionPromptModel,
    LoraModel,
    IA3Model,
    LoKrModel,
    # LoraConfig,
    PromptEmbedding,
    MultitaskPromptEmbedding,
    PromptEncoder,
    PrefixEncoder,
    LoHaModel,
    PolyModel,
    LNTuningModel,
)
from .utils import (
    # SAFETENSORS_WEIGHTS_NAME,
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    WEIGHTS_NAME,
    PeftType,
    TaskType,
    _prepare_prompt_learning_config,
    # _set_adapter,
    _set_trainable,
    # add_library_to_model_card,
    get_peft_model_state_dict,
    load_peft_weights,
    set_peft_model_state_dict,
    shift_tokens_right,
    _get_batch_size, # will be used for prompt learning methods
)


PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.LORA: LoraModel,
    PeftType.ADAPTION_PROMPT: AdaptionPromptModel,
    PeftType.IA3: IA3Model,
    PeftType.ADALORA: AdaLoraModel,
    PeftType.LOKR: LoKrModel,
    PeftType.LOHA: LoHaModel,
    PeftType.POLY: PolyModel,
    PeftType.LN_TUNING: LNTuningModel,
}

class PeftModel(nn.Module):
    """
    Base model encompassing various Peft methods.

    Args:
        model ([`~mindnlp.models.PreTrainedModel`]): The base transformer model used for Peft.
        peft_config ([`PeftConfig`]): The configuration of the Peft model.
    """
    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        r"""
        __init__

        This method initializes an instance of the PeftModel class.

        Args:
            self: The instance of the PeftModel class.
            model: The base model used for the PeftModel instance.
            peft_config (PeftConfig): An instance of PeftConfig class containing configuration for the PEFT (Prompt-based Entity Fine-Tuning) process.
            adapter_name (str, optional): The name of the adapter being used. Defaults to 'default'.

        Returns:
            None. This method does not return any value.

        Raises:
            - TypeError: If the provided model is not of the expected type.
            - ValueError: If the provided peft_config is not valid or does not contain necessary information.
            - KeyError: If there is an issue with accessing or setting attributes.
        """
        super().__init__()
        self.base_model = model
        self.config = getattr(self.base_model, "config", {"model_type": "custom"})
        self.cells_to_save = None
        self.peft_config: Dict[str, PeftConfig] = {}
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        self.base_model_dtype = getattr(model, "dtype", None)
        self.special_peft_forward_args = {"adapter_name"}
        if not peft_config.is_prompt_learning:
            self.peft_config[adapter_name] = peft_config
            self.base_model = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type](
                self.base_model, self.peft_config, adapter_name
            )
            self.set_additional_trainable_cells(peft_config, adapter_name)
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
        from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING
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

    def _setup_prompt_encoder(self, adapter_name: str):
        config = self.peft_config[adapter_name]
        if not hasattr(self, "prompt_encoder"):
            self.prompt_encoder = nn.ModuleDict({})
            self.prompt_tokens = {}
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                # Make sure to freeze Tranformers model
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name
        if transformer_backbone is None:
            transformer_backbone = self.base_model

        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1

        for named_param, value in list(transformer_backbone.named_parameters()):
            # for ZeRO-3, the tensor is sharded across accelerators and deepspeed modifies it to a tensor with shape [0]
            # the actual unsharded shape is stored in "ds_shape" attribute
            # special handling is needed in case the model is initialized in deepspeed.zero.Init() context or HfDeepSpeedConfig
            # has been called before
            # For reference refer to issue: https://github.com/huggingface/peft/issues/996

            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        if config.peft_type == PeftType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(config, self.word_embeddings)
        elif config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            prompt_encoder = MultitaskPromptEmbedding(config, self.word_embeddings)
        elif config.peft_type == PeftType.P_TUNING:
            prompt_encoder = PromptEncoder(config)
        elif config.peft_type == PeftType.PREFIX_TUNING:
            prompt_encoder = PrefixEncoder(config)
        else:
            raise ValueError("Not supported")

        self.prompt_encoder.update(nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = ops.arange(
            config.num_virtual_tokens * config.num_transformer_submodules
        ).long()

    def load_adapter(self, model_id: str, adapter_name: str, is_trainable: bool = False, **kwargs):
        """load adapter to peft model, called by `model.from_pretrained`."""
        # NOTE: remove download logic.
        if adapter_name not in self.peft_config:
            raise ValueError(f"{adapter_name} is not a valid adapter name. Valid names: {self.peft_config.keys()}")

        adapters_weights = load_peft_weights(model_id)

        # load the weights into the model
        load_result = set_peft_model_state_dict(self, adapters_weights, adapter_name=adapter_name)
        # TODO: add parallel logic & offload logic & device map logic(dispatch_model)

        # Set model in evaluation mode to deactivate Dropout cells by default
        if not is_trainable:
            self.set_train(False)

        return load_result

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif not hasattr(param, "quant_storage"):
                    num_bytes = 1
                else:
                    num_bytes = param.quant_storage.itemsize
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def get_prompt_embedding_to_save(self, adapter_name: str) -> mindspore.Tensor:
        """
        Returns the prompt embedding to save when saving the model. Only applicable when using a prompt learning
        method.
        """
        prompt_encoder = self.prompt_encoder[adapter_name]
        prompt_tokens = (
            self.prompt_tokens[adapter_name].unsqueeze(0).broadcast_to((1, -1))
        )
        if self.peft_config[adapter_name].peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : self.peft_config[adapter_name].num_virtual_tokens]

        if self.peft_config[adapter_name].peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            prompt_embeddings = super(MultitaskPromptEmbedding, prompt_encoder).forward(prompt_tokens) # pylint: disable=bad-super-call
        else:
            prompt_embeddings = prompt_encoder(prompt_tokens)

        embedding = prompt_embeddings[0]
        return Tensor(embedding.asnumpy())

    def get_prompt(self, batch_size: int, task_ids: Optional[mindspore.Tensor] = None) -> mindspore.Tensor:
        """
        Returns the virtual prompts to use for Peft. Only applicable when using a prompt learning method.
        """
        peft_config = self.active_peft_config
        prompt_encoder = self.prompt_encoder[self.active_adapter]
        prompt_tokens = (
            self.prompt_tokens[self.active_adapter]
            .unsqueeze(0)
            .broadcast_to((batch_size, -1))
        )
        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]
            if peft_config.inference_mode:
                past_key_values = prompt_encoder.embedding.weight.tile((batch_size, 1, 1))
            else:
                past_key_values = prompt_encoder(prompt_tokens)
            if self.base_model_dtype is not None:
                past_key_values = past_key_values.to(self.base_model_dtype)
            past_key_values = past_key_values.view(
                batch_size,
                peft_config.num_virtual_tokens,
                peft_config.num_layers * 2,
                peft_config.num_attention_heads,
                peft_config.token_dim // peft_config.num_attention_heads,
            )
            if peft_config.num_transformer_submodules == 2:
                past_key_values = ops.cat([past_key_values, past_key_values], dim=2)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
                peft_config.num_transformer_submodules * 2
            )
            if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
                post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
                past_key_values = post_process_fn(past_key_values)
            return past_key_values
        else:
            if peft_config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
                prompts = prompt_encoder(prompt_tokens, task_ids)
            else:
                if peft_config.inference_mode:
                    prompts = prompt_encoder.embedding.weight.tile((batch_size, 1, 1))
                else:
                    prompts = prompt_encoder(prompt_tokens)
            return prompts

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped cell."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        """
        # print(self.get_base_model().layers[0].__class__.forward)
        return self.get_base_model()(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.get_base_model().generate(*args, **kwargs)

    @contextmanager
    def disable_adapter(self):
        """
        Disables the adapter cell.
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
        return (
            self.base_model
            if self.active_peft_config.is_prompt_learning
            or self.peft_type == PeftType.POLY
            else self.base_model.model
        )

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

        self.set_additional_trainable_cells(peft_config, adapter_name)

    def set_additional_trainable_cells(self, peft_config, adapter_name):
        """set additional trainable cells"""
        if getattr(peft_config, "cells_to_save", None) is not None:
            if self.cells_to_save is None:
                self.cells_to_save = set(peft_config.cells_to_save)
            else:
                self.cells_to_save.update(peft_config.cells_to_save)
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
        """
        Initializes a new instance of the PeftModelForSequenceClassification class.

        Args:
            self: The instance of the PeftModelForSequenceClassification class.
            model: The base model to be used for sequence classification (e.g., a pre-trained language model).
            peft_config (PeftConfig): The configuration for the PEFT (Probing and Evaluation for Transformers) model.
            adapter_name (str, optional): The name of the adapter to be used. Defaults to 'default'.

        Returns:
            None. This method initializes the instance with the specified parameters.

        Raises:
            None.
        """
        super().__init__(model, peft_config, adapter_name)
        if self.cells_to_save is None:
            self.cells_to_save = {"classifier", "score"}
        else:
            self.cells_to_save.update({"classifier", "score"})

        for name, _ in self.base_model.cells_and_names():
            if any(cell_name in name for cell_name in self.cells_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self, adapter_name)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        """
        Forward pass of the model.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            # NOTE:some args not exists in base model
            # inputs_embeds=inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,

            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = ops.ones(batch_size, peft_config.num_virtual_tokens, dtype=attention_mask.dtype)
            attention_mask = ops.cat((prefix_attention_mask, attention_mask), dim=1)
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

        # if peft_config.peft_type == PeftType.PREFIX_TUNING:
        #     return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        if kwargs.get("token_type_ids", None) is not None:
            kwargs["token_type_ids"] = ops.cat(
                (
                    ops.zeros(batch_size, peft_config.num_virtual_tokens, dtype=kwargs["token_type_ids"].dtype),
                    kwargs["token_type_ids"],
                ),
                dim=1,
            )
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        prompts = self.get_prompt(batch_size=batch_size)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = ops.cat((prompts, inputs_embeds), dim=1)
        return self.base_model(inputs_embeds=inputs_embeds, **kwargs)


class PeftModelForCausalLM(PeftModel):
    """
    Peft model for causal language modeling.

    Args:
        model ([`~mindnlp.models.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.
    """
    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        r"""
        Initializes a new instance of the PeftModelForCausalLM class.

        Args:
            self: The instance itself.
            model: The underlying model for the adapter.
            peft_config (PeftConfig): The configuration for the PEFT (Plug and Fine-tune) adapter.
            adapter_name (str): The name of the adapter. Defaults to 'default'.

        Returns:
            None. This method does not return any value.

        Raises:
            N/A
        """
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        """
        Forward pass of the model.
        """
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
            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids
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
            attention_mask = ops.cat((prefix_attention_mask, attention_mask), dim=1)

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
            prefix_labels = ops.full((batch_size, peft_config.num_virtual_tokens), -100)
            kwargs["labels"] = ops.cat((prefix_labels, labels), dim=1)
        prompts = self.get_prompt(batch_size=batch_size)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = ops.cat((prompts, inputs_embeds), dim=1)
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

    def prepare_inputs_for_generation(self, *args, task_ids: Optional[mindspore.Tensor] = None, **kwargs,):
        """prepare_inputs_for_generation."""
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if peft_config.peft_type == PeftType.POLY:
            model_kwargs["task_ids"] = task_ids
        if isinstance(peft_config, PromptLearningConfig):
            if model_kwargs.get("attention_mask", None) is not None:
                prefix_attention_mask = ops.ones(
                    model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens)
                model_kwargs["attention_mask"] = ops.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
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
                    model_kwargs["inputs_embeds"] = ops.cat((prompts, inputs_embeds), dim=1)
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
        r"""
        Initialize a new PeftModelForSeq2SeqLM object.

        Args:
            self: The instance of the PeftModelForSeq2SeqLM class.
            model: The model to be used for the PeftModelForSeq2SeqLM.
            peft_config (PeftConfig): The configuration object for the PeftModelForSeq2SeqLM.
            adapter_name (str): The name of the adapter to be used, defaults to 'default'.

        Returns:
            None. This method initializes the PeftModelForSeq2SeqLM object.

        Raises:
            None.
        """
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )

    def forward(
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
        task_ids=None,
        **kwargs,
    ):
        """
        Forward pass of the model.
        """
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
            prefix_attention_mask = ops.ones(batch_size, peft_config.num_virtual_tokens)
            decoder_attention_mask = ops.cat((prefix_attention_mask, decoder_attention_mask), dim=1)

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
                prefix_attention_mask = ops.ones(batch_size, peft_config.num_virtual_tokens)
                kwargs["attention_mask"] = ops.cat((prefix_attention_mask, attention_mask), dim=1)

            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = ops.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)

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
                prefix_attention_mask = ops.ones(batch_size, peft_config.num_virtual_tokens, dtype=attention_mask.dtype)
                kwargs["attention_mask"] = ops.cat((prefix_attention_mask, attention_mask), dim=1)
            # concat prompt labels
            if labels is not None:
                if peft_config.num_transformer_submodules == 1:
                    kwargs["labels"] = labels
                elif peft_config.num_transformer_submodules == 2:
                    prefix_labels = ops.full((batch_size, peft_config.num_virtual_tokens), -100)
                    kwargs["labels"] = ops.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = ops.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)
            if peft_config.num_transformer_submodules == 1:
                return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
            elif peft_config.num_transformer_submodules == 2:
                decoder_inputs_embeds = ops.cat(
                    (prompts[:, peft_config.num_virtual_tokens :], decoder_inputs_embeds), dim=1
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

                    inputs_embeds = ops.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)
                    kwargs["inputs_embeds"] = inputs_embeds

                    if "attention_mask" in kwargs:
                        prefix_attention_mask = ops.ones(batch_size, peft_config.num_virtual_tokens)
                        kwargs["attention_mask"] = ops.cat((prefix_attention_mask, kwargs["attention_mask"]), dim=1)

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

    def prepare_inputs_for_generation(self, *args, task_ids: mindspore.Tensor = None, **kwargs):
        """prepare inputs for generation"""
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if peft_config.peft_type == PeftType.POLY:
            model_kwargs["task_ids"] = task_ids
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
        r"""
        Initializes a new instance of the PeftModelForTokenClassification class.

        Args:
            self: The instance of the PeftModelForTokenClassification class.
            model: The model used for token classification.
            peft_config (PeftConfig, optional): The configuration for the Peft model. Defaults to None.
            adapter_name (str, optional): The name of the adapter. Defaults to 'default'.

        Returns:
            None. This method does not return a value.

        Raises:
            N/A
        """
        super().__init__(model, peft_config, adapter_name)
        if self.cells_to_save is None:
            self.cells_to_save = {"classifier", "score"}
        else:
            self.cells_to_save.update({"classifier", "score"})

        for name, _ in self.base_model.cells_and_names():
            if any(cell_name in name for cell_name in self.cells_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self, adapter_name)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        """
        Forward pass of the model.
        """
        peft_config = self.active_peft_config
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not isinstance(peft_config, PromptLearningConfig):
            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids
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
            attention_mask = ops.cat((prefix_attention_mask, attention_mask), dim=1)
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
                        ops.zeros(batch_size, peft_config.num_virtual_tokens),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = ops.cat((prompts, inputs_embeds), dim=1)
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
        r"""
        Performs the forward pass for the prefix tuning task in the PeftModelForTokenClassification class.

        Args:
            self (PeftModelForTokenClassification): The instance of the PeftModelForTokenClassification class.
            input_ids (torch.Tensor): The input token IDs tensor of shape [batch_size, sequence_length].
            attention_mask (torch.Tensor): The attention mask tensor of shape [batch_size, sequence_length].
            inputs_embeds (torch.Tensor): The input embeddings tensor of shape [batch_size, sequence_length, hidden_size].
            labels (torch.Tensor): The labels tensor of shape [batch_size, sequence_length].
            output_attentions (bool): Whether to output attentions. Defaults to None.
            output_hidden_states (bool): Whether to output hidden states. Defaults to None.
            return_dict (bool): Whether to return a dictionary. Defaults to None.

        Returns:
            None: This method does not return any value. Instead, it updates the internal state of the model.

        Raises:
            ValueError: If the model does not support past key values which are required for prefix tuning.

        """
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
            transformer_backbone_name = self.base_model.get_subcell(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            sequence_output = outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.cells_and_names())]:
                sequence_output = self.base_model.dropout(sequence_output)
            logits = self.base_model.get_subcell(self.cls_layer_name)(sequence_output)

            loss = None
            if labels is not None:
                loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
