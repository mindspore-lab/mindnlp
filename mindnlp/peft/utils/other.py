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
import mindspore
from mindspore import ops, Parameter
from mindspore.common.initializer import initializer, Normal
from mindnlp._legacy.nn import Matmul

class ModulesToSaveWrapper(mindspore.nn.Cell):
    """
    save module
    """
    def __init__(self, module_to_save, adapter_name):
        super().__init__()
        self.original_module = module_to_save
        self.modules_to_save = mindspore.nn.CellList()
        self.update(adapter_name)
        self.active_adapter = adapter_name

    def construct(self, *args, **kwargs):
        if self.active_adapter not in self.modules_to_save:
            return self.original_module(*args, **kwargs)
        return self.modules_to_save[self.active_adapter](*args, **kwargs)

def _get_submodules(model, key):
    """
    get submodules
    """
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def _set_trainable(model, adapter_name):
    """
    set trainable
    """
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        target_module_found = any(key.endswith(target_key) for target_key in model.modules_to_save)
        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)

            if isinstance(target, ModulesToSaveWrapper):
                # 判断是否是此数据类型
                target.update(adapter_name)
            else:
                for param in target.parameters():
                    param.requires_grad = True
                setattr(parent, target_name, ModulesToSaveWrapper(target, adapter_name))


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


def transpose(weight, fan_in_fan_out):
    """
    transpose weight
    """
    return weight.T if fan_in_fan_out else weight


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

TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "opt": ["q_proj", "v_proj"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "deberta": ["in_proj"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"]
}

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

WEIGHTS_NAME = "adapter_model.bin"
CONFIG_NAME = "adapter_config.json"
