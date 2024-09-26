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
"""save and load"""
import os
from collections import OrderedDict

import mindspore

from .peft_types import PeftType
from .constants import WEIGHTS_NAME

def get_data_list(param_dict):
    """Get state dict of the Peft model for saving."""
    data_list = OrderedDict()  # {key: [dims, tensor_type, data]}

    for key, value in param_dict.items():
        data_list[key] = []
        dims = []
        if value.shape == ():
            dims.append(0)
        else:
            for dim in value.shape:
                dims.append(dim)
        data_list[key].append(dims)
        tensor_type = str(value.dtype)
        data_list[key].append(tensor_type)
        data = value.asnumpy().reshape(-1)
        data_list[key].append(data)

    return data_list


def get_peft_model_state_dict(model, state_dict=None, adapter_name="default"):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
    """
    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = get_data_list(model.parameters_dict())
    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {k: v for k, v in to_return.items() if (("lora_" in k and adapter_name in k) or ("bias" in k))}
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                rank_pattern = {k.replace(f".{adapter_name}", ""): v for k, v in rank_pattern.items()}
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(rank_pattern, to_return, adapter_name)
    elif config.peft_type == PeftType.ADAPTION_PROMPT:
        to_return = {k: state_dict[k] for k in state_dict if k.split(".")[-1].startswith("adaption_")}
    elif config.peft_type == PeftType.IA3:
        to_return = {k: state_dict[k] for k in state_dict if "ia3_" in k}
    elif config.peft_type == PeftType.LOKR:
        to_return = {k: state_dict[k] for k in state_dict if "lokr_" in k}
    elif config.peft_type == PeftType.POLY:
        to_return = {k: state_dict[k] for k in state_dict if "poly_" in k}
    elif config.peft_type == PeftType.LN_TUNING:
        to_return = {k: state_dict[k] for k in state_dict if "ln_tuning_" in k}
    elif config.peft_type == PeftType.LOHA:
        to_return = {k: state_dict[k] for k in state_dict if "hada_" in k}
    elif config.is_prompt_learning:
        to_return = {}
        if config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            to_return["prefix_task_cols"] = model.prompt_encoder[adapter_name].prefix_task_cols
            to_return["prefix_task_rows"] = model.prompt_encoder[adapter_name].prefix_task_rows
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        else:
            if config.inference_mode:
                prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
            else:
                prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
        to_return["prompt_embeddings"] = prompt_embeddings
        to_return = get_data_list(to_return)
    else:
        raise NotImplementedError

    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(f"{cell_name}.modules_to_save.{adapter_name}" in key for cell_name in model.modules_to_save):
                to_return[key.replace("modules_to_save.", "")] = value

    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    return to_return


def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name="default"):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    strict_load = False
    if model.modules_to_save is not None:
        for key, value in peft_model_state_dict.items():
            if any(cell_name in key for cell_name in model.modules_to_save):
                for cell_name in model.modules_to_save:
                    if cell_name in key:
                        key = key.replace(cell_name, f"{cell_name}.modules_to_save.{adapter_name}")
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    if config.peft_type in (
        PeftType.LORA,
        PeftType.IA3,
        PeftType.ADALORA,
        PeftType.LOKR,
        PeftType.LOHA,
        PeftType.POLY,
        PeftType.LN_TUNING,
    ):
        peft_model_state_dict = {}
        parameter_prefix = {
            PeftType.IA3: "ia3_",
            PeftType.LORA: "lora_",
            PeftType.ADALORA: "lora_",
            PeftType.LOKR: "lokr_",
            PeftType.LOHA: "hada_",
            PeftType.POLY: "poly_",
            PeftType.LN_TUNING: "ln_tuning_",
        }[config.peft_type]
        for k, v in state_dict.items():
            if parameter_prefix in k:
                suffix = k.split(parameter_prefix)[1]
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
                else:
                    k = f"{k}.{adapter_name}"
                peft_model_state_dict[k] = v
            else:
                peft_model_state_dict[k] = v
        if config.peft_type == PeftType.ADALORA:
            strict_load = True
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_cells_by_rank_pattern(rank_pattern, adapter_name)
    elif config.is_prompt_learning or config.peft_type == PeftType.ADAPTION_PROMPT:
        peft_model_state_dict = state_dict
    else:
        raise NotImplementedError

    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
    if config.is_prompt_learning:
        model.prompt_encoder[adapter_name].embedding.load_state_dict(
            {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
        )

    if config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
        model.prompt_encoder[adapter_name].load_state_dict(peft_model_state_dict, strict=False)
    return load_result


def load_peft_weights(model_id: str,) -> dict:
    r"""
    A helper method to load the PEFT weights from local storage. Add download logic later.

    Args:
        model_id (`str`):
            The local path to the adapter weights or the name of the adapter to load from the HuggingFace Hub.
    """
    path = model_id

    filename = os.path.join(path, WEIGHTS_NAME)
    if not os.path.exists(filename):
        # TODO: add download logic later
        raise ValueError(f"load peft model failed, peft model file: {filename} not exists.")

    adapters_weights = mindspore.load_checkpoint(filename)

    return adapters_weights
