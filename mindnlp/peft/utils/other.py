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
# pylint: disable=C0103
"""other utils"""
import copy
from typing import Optional, List

import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp._legacy.nn import Matmul
from mindnlp.abc import CellDict

def _get_batch_size(input_ids: Optional[Tensor], inputs_embeds: Optional[Tensor]) -> int:
    """Get the batch size based on either input_ids or input_embeds

    Raises an ValueError if both are None.

    """
    if (input_ids is None) and (inputs_embeds is None):
        raise ValueError("You have to provide either input_ids or inputs_embeds")

    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]
    return batch_size

class ModulesToSaveWrapper(mindspore.nn.Cell):
    """
    save module
    """
    def __init__(self, module_to_save, adapter_name):
        super().__init__()
        self.original_module = module_to_save
        self.modules_to_save = CellDict()
        self.update(adapter_name)
        self.active_adapter = adapter_name
        self.disable_adapters = False

    def update(self, adapter_name):
        """
        update modules_to_save.
        """
        self.modules_to_save.update({adapter_name: copy.deepcopy(self.original_module)})

    #     if hasattr(self.modules_to_save[adapter_name], "_hf_hook"):
    #         old_hook = self.modules_to_save[adapter_name]._hf_hook
    #         new_hook = self._create_new_hook(old_hook)
    #         remove_hook_from_module(self.modules_to_save[adapter_name])
    #         add_hook_to_module(self.modules_to_save[adapter_name], new_hook)

    # def _create_new_hook(self, old_hook):
    #     r"""
    #     Creates a new hook based on the old hook. Use it only if you know what you are doing !
    #     """
    #     old_hook_cls = getattr(accelerate.hooks, old_hook.__class__.__name__)
    #     old_hook_attr = old_hook.__dict__
    #     filtered_old_hook_attr = {}
    #     old_hook_init_signature = inspect.signature(old_hook_cls.__init__)
    #     for k in old_hook_attr.keys():
    #         if k in old_hook_init_signature.parameters:
    #             filtered_old_hook_attr[k] = old_hook_attr[k]
    #     new_hook = old_hook_cls(**filtered_old_hook_attr)
    #     return new_hook

    def construct(self, X):
        # TODO:*args, **kwargs
        if self.disable_adapters or (self.active_adapter not in self.modules_to_save.keys()):
            return self.original_module(X)
        return self.modules_to_save[self.active_adapter](X)

def custom_get_submodule(model: mindspore.nn.Cell, target: str) -> mindspore.nn.Cell:
    """
    Returns the submodule given by ``target`` if it exists, otherwise throws an error.
    功能和 torch.nn.Module 相似
    """
    if target == "":
        return model

    atoms: List[str] = target.split(".")
    mod: mindspore.nn.Cell = model

    for item in atoms:
        if not hasattr(mod, item):
            raise AttributeError(mod + " has no attribute `" + item + "`")

        mod = getattr(mod, item)

        if not isinstance(mod, mindspore.nn.Cell):
            raise AttributeError("`" + item + "` is not an nn.Module")

    return mod

def _get_submodules(model, key):
    """
    get submodules
    """
    parent_key = ".".join(key.split(".")[:-1])
    parent = custom_get_submodule(model, parent_key)
    target_name = key.split(".")[-1]
    target = custom_get_submodule(model, key)

    return parent, target, target_name


def _set_trainable(model, adapter_name):
    """
    set trainable
    """
    key_list = [key for key, _ in model.cells_and_names()]  # named_modules cells_and_names
    for key in key_list:
        target_module_found = any(key.endswith(target_key) for target_key in model.modules_to_save)
        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)

            if isinstance(target, ModulesToSaveWrapper):
                # 判断是否是此数据类型
                target.update(adapter_name)
            else:
                for _, param in target.parameters_and_names():
                    param.requires_grad = True
                warp_cell = ModulesToSaveWrapper(target, adapter_name)
                # parent[int(target_name)] = warp_cell
                setattr(parent, target_name, warp_cell)

                # TODO:the implemtation of mindspore, __setitem__ is not consistent with __setattr__ here.
                # self.cell_list is not set correctly if __setattr__'s value type is SequentialCell.
                # Thus we set it apparently here. This line may be removed later.
                if isinstance(parent, nn.SequentialCell):
                    parent.cell_list = list(parent._cells.values())



def _freeze_adapter(model, adapter_name):
    """
    freeze adapter
    """
    for n, p in model.named_parameters():
        if adapter_name in n:
            p.requires_grad = False


def _set_adapter(model, adapter_name):
    for module in model.modules():
        if isinstance(module, ModulesToSaveWrapper):
            module.active_adapter = adapter_name


def _prepare_prompt_learning_config(peft_config, model_config):
    if peft_config.num_layers is None:
        if "num_hidden_layers" in model_config:
            num_layers = model_config["num_hidden_layers"]
        elif "num_layers" in model_config:
            num_layers = model_config["num_layers"]
        elif "n_layer" in model_config:
            num_layers = model_config["n_layer"]
        else:
            raise ValueError("Please specify `num_layers` in `peft_config`")
        peft_config.num_layers = num_layers

    if peft_config.token_dim is None:
        if "hidden_size" in model_config:
            token_dim = model_config["hidden_size"]
        elif "n_embd" in model_config:
            token_dim = model_config["n_embd"]
        elif "d_model" in model_config:
            token_dim = model_config["d_model"]
        else:
            raise ValueError("Please specify `token_dim` in `peft_config`")
        peft_config.token_dim = token_dim

    if peft_config.num_attention_heads is None:
        if "num_attention_heads" in model_config:
            num_attention_heads = model_config["num_attention_heads"]
        elif "n_head" in model_config:
            num_attention_heads = model_config["n_head"]
        elif "num_heads" in model_config:
            num_attention_heads = model_config["num_heads"]
        elif "encoder_attention_heads" in model_config:
            num_attention_heads = model_config["encoder_attention_heads"]
        else:
            raise ValueError("Please specify `num_attention_heads` in `peft_config`")
        peft_config.num_attention_heads = num_attention_heads

    if getattr(peft_config, "encoder_hidden_size", None) is None:
        setattr(peft_config, "encoder_hidden_size", peft_config.token_dim)

    return peft_config


def transpose(weight, fan_in_fan_out):
    """
    transpose weight
    """
    return weight.T if fan_in_fan_out else weight
    # return ops.transpose(weight, input_perm=?) if fan_in_fan_out else weight


def shift_tokens_right(input_ids: mindspore.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`): input ids
        pad_token_id (`int`): The id of the `padding` token.
        decoder_start_token_id (`int`): The id of the `start` token.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class Conv1D(mindspore.nn.Cell):
    """
    1D-convolutional layer Basically works like a linear layer but the weights are transposed.

    Args:
        n_out (`int`): The number of output features.
        n_in (`int`): The number of input features.
    """

    def __init__(self, n_out, n_in):
        super().__init__()
        self.n_out = n_out
        self.weight = Parameter(initializer(Normal(sigma=0.02), (n_in, n_out), mindspore.float32))
        self.bias = Parameter(ops.zeros(n_out, mindspore.float32))
        self.matmul = Matmul()

    def construct(self, x):
        size_out = x.shape[:-1] + (self.n_out,)
        x = self.matmul(x.view(-1, x.shape[-1]), self.weight) + self.bias
        x = x.view(size_out)
        return x


# Target_modules mapping (model -> qkv), which is highly related to **Mindnlp model** implementation.
# lora
TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    # "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    # "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    # "gptj": ["q_proj", "v_proj"],
    # "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    # "xlm-roberta": ["query", "value"],
    # "electra": ["query", "value"],
    # "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    # "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
    # "gpt_bigcode": ["c_attn"],
    # "mpt": ["Wqkv"],
    # "RefinedWebModel": ["query_key_value"],
    # "RefinedWeb": ["query_key_value"],
    "falcon": ["query_key_value"],
    # "btlm": ["c_proj", "c_attn"],
    # "codegen": ["qkv_proj"],
}

TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING = {}
TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING = {}
COMMON_LAYERS_PATTERN = ["layers", "h", "block", "blocks", "layer"]
TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING = {}


WEIGHTS_NAME = "adapter_model.ckpt"
CONFIG_NAME = "adapter_config.json"

CLAMP_QUANTILE = 0.99
