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
"""other utils"""
import copy
from contextlib import nullcontext
from typing import Optional, List

import mindspore
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Normal

from mindnlp.core import nn, ops
from mindnlp.core.nn import ParameterDict

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

class ModulesToSaveWrapper(nn.Module):

    r"""
    This class represents a wrapper for saving and managing modules in a neural network. It provides functionality to save and switch between different modules, known as adapters, while also maintaining the
original module for reference. The class includes methods for enabling and disabling adapters, setting the active adapter, updating the saved modules, and forwarding the model with the appropriate adapter.
    
    The class inherits from nn.Module and includes the following methods:
    - __init__: Initializes the ModulesToSaveWrapper instance with the original cell to save and the initial adapter name.
    - check_cell: Performs sanity checks on the original cell to ensure compatibility with the saving mechanism.
    - disable_adapters: Toggles the enabling and disabling of adapters, managing the requires_grad flag for adapter weights.
    - active_adapter: Returns the name of the currently active adapter.
    - weight: Retrieves the weight of the original cell or the active adapter's cell if available.
    - update: Updates the saved cells with a new adapter, creating a deep copy of the original cell.
    - forward: Constructs the model using the original cell or the active adapter's cell based on the adapter status.
    - enable_adapters: Toggles the enabling and disabling of adapters, managing the requires_grad flag for adapter weights.
    - set_adapter: Sets the active adapter, making it trainable and updating the requires_grad flag for the cells.
    
    The class provides a flexible way to manage and switch between different modules within a neural network.
    """
    def __init__(self, cell_to_save, adapter_name):
        r"""
        Initializes an instance of the ModulesToSaveWrapper class.
        
        Args:
            self (ModulesToSaveWrapper): The current instance of the class.
            cell_to_save (Any): The cell to be saved.
            adapter_name (str): The name of the adapter.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        """
        super().__init__()
        self.original_cell = cell_to_save
        self.cells_to_save = nn.ModuleDict({})
        self._active_adapter = adapter_name
        self._disable_adapters = False
        self.update(adapter_name)
        self.check_cell()

    def check_cell(self):
        """Perform some sanity checks on the cell to ensure that it works"""
        # Try to anticipate some cells that users could try to target that would not work.
        # Note: It's not possible to check hasattr(cell, "forward"), since that returns True for ModuleDict and
        # ModuleList, even though their forward methods cannot be called
        forbidden_classes = (nn.ModuleDict, nn.ModuleList, mindspore.ParameterTuple, ParameterDict)
        if isinstance(self.original_cell, forbidden_classes):
            cls_name = self.original_cell.__class__.__name__
            raise TypeError(f"cells_to_save cannot be applied to cells of type {cls_name}")

    @property
    def disable_adapters(self) -> bool:
        r"""
        Method to retrieve the status of whether adapters are disabled in the ModulesToSaveWrapper class.
        
        Args:
            self (ModulesToSaveWrapper): The instance of the ModulesToSaveWrapper class.
                This parameter is always required as it refers to the instance calling this method.
        
        Returns:
            bool: A boolean value indicating whether adapters are disabled.
                Returns True if adapters are disabled, False otherwise. 
        
        Raises:
            No specific exceptions are raised by this method.
        """
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def active_adapter(self) -> str:
        r"""
        This method retrieves the active adapter from the ModulesToSaveWrapper class.
        
        Args:
            self: The instance of the ModulesToSaveWrapper class.
        
        Returns:
            str: The active adapter as a string.
        
        Raises:
            None
        """
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    @property
    def weight(self):
        r"""
        This method 'weight' is a property method within the 'ModulesToSaveWrapper' class.
        
        Args:
            - self: (object) The instance of the 'ModulesToSaveWrapper' class.
        
        Returns:
            - None: This method returns a value of type None.
        
        Raises:
            - None: This method does not raise any exceptions.
        """
        if self.active_adapter not in self.cells_to_save:
            return self.original_cell.weight
        return self.cells_to_save[self.active_adapter].weight

    def update(self, adapter_name):
        r"""
        Updates the ModulesToSaveWrapper with a new adapter.
        
        Args:
            self (ModulesToSaveWrapper): The instance of ModulesToSaveWrapper.
            adapter_name (str): The name of the adapter to update.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            - AttributeError: If the 'cells_to_save' attribute does not contain the specified 'adapter_name'.
            - RuntimeError: If an error occurs during the update process.
            - ValueError: If the 'adapter_name' parameter is not a string.
        """
        context_manager = nullcontext()
        for _, param in self.original_cell.parameters_and_names():
            num_params = param.numel()

        with context_manager:
            self.cells_to_save.update(nn.ModuleDict({adapter_name: copy.deepcopy(self.original_cell)}))

        if hasattr(self.cells_to_save[adapter_name], "_hf_hook"):
            old_hook = self.cells_to_save[adapter_name]._hf_hook
            new_hook = self._create_new_hook(old_hook)
            # remove_hook_from_cell(self.cells_to_save[adapter_name])
            # add_hook_to_cell(self.cells_to_save[adapter_name], new_hook)

        self.original_cell.requires_grad_(False)
        if adapter_name == self.active_adapter:
            self.cells_to_save[adapter_name].requires_grad_(True)

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

    def forward(self, *args, **kwargs):
        r"""
        This method forwards and returns the appropriate cell based on the active adapter within the ModulesToSaveWrapper class.
        
        Args:
            self: An instance of the ModulesToSaveWrapper class.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            - N/A
        """
        if self.disable_adapters or (self.active_adapter not in self.cells_to_save):
            return self.original_cell(*args, **kwargs)
        return self.cells_to_save[self.active_adapter](*args, **kwargs)

    def enable_adapters(self, enabled: bool):
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if self._disable_adapters is not enabled:
            # already in the desired state, do nothing
            return

        if enabled:
            self.original_cell.requires_grad_(False)
            self.cells_to_save[self.active_adapter].requires_grad_(True)
            self._disable_adapters = False
        else:
            self.original_cell.requires_grad_(True)
            self.cells_to_save.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_name: str):
        """Set the active adapter

        Additionally, this function will set the specified adapter to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (str): The name of the adapter to set as active
        """
        if adapter_name not in self.cells_to_save:
            raise ValueError(f"Adapter {adapter_name} not found in {self.cells_to_save.keys()}")

        self.cells_to_save[self.active_adapter].requires_grad_(False)
        self.cells_to_save[adapter_name].requires_grad_(True)
        self._active_adapter = adapter_name


def custom_get_subcell(model: nn.Module, target: str) -> nn.Module:
    """
    Returns the subcell given by ``target`` if it exists, otherwise throws an error.
    功能和 torch.nn.Module 相似
    """
    if target == "":
        return model

    atoms: List[str] = target.split(".")
    mod: nn.Module = model

    for item in atoms:
        if not hasattr(mod, item):
            raise AttributeError(mod + " has no attribute `" + item + "`")

        mod = getattr(mod, item)

        if not isinstance(mod, nn.Module):
            raise AttributeError("`" + item + "` is not an nn.Module")

    return mod

def _get_subcells(model, key):
    """
    get subcells
    """
    parent_key = ".".join(key.split(".")[:-1])
    parent = custom_get_subcell(model, parent_key)
    target_name = key.split(".")[-1]
    target = custom_get_subcell(model, key)

    return parent, target, target_name


def _set_trainable(model, adapter_name):
    """
    set trainable
    """
    key_list = [key for key, _ in model.cells_and_names()]  # named_cells cells_and_names
    for key in key_list:
        target_cell_found = any(key.endswith(target_key) for target_key in model.cells_to_save)
        if target_cell_found:
            parent, target, target_name = _get_subcells(model, key)

            if isinstance(target, ModulesToSaveWrapper):
                target.update(adapter_name)
            else:
                for _, param in target.parameters_and_names():
                    param.requires_grad = True
                warp_cell = ModulesToSaveWrapper(target, adapter_name)
                # parent[int(target_name)] = warp_cell
                setattr(parent, target_name, warp_cell)

                # TODO:the implemtation of mindspore, __setitem__ is not consistent with __setattr__ here.
                # self.cell_list is not set correctly if __setattr__'s value type is Sequential.
                # Thus we set it apparently here. This line may be removed later.
                if isinstance(parent, nn.Sequential):
                    parent.cell_list = list(parent._cells.values())

    for n, p in model.parameters_and_names():
        if n != p.name:
            p.name = n


def _freeze_adapter(model, adapter_name):
    """
    freeze adapter
    """
    for n, p in model.parameters_and_names():
        if adapter_name in n:
            p.requires_grad = False


def _set_adapter(model, adapter_name):
    r"""
    Sets the active adapter for the given model.
    
    Args:
        model (object): The model for which the active adapter needs to be set.
        adapter_name (str): The name of the adapter to be set as active.
    
    Returns:
        None. This function does not return any value.
    
    Raises:
        None. This function does not raise any exceptions.
    """
    for cell in model.cells():
        if isinstance(cell, ModulesToSaveWrapper):
            cell.active_adapter = adapter_name


def _prepare_prompt_learning_config(peft_config, model_config):
    r"""
    Args:
        peft_config (object): The PEFT configuration object containing the parameters for prompt learning.
        model_config (dict): The model configuration dictionary containing the parameters for the underlying model.
    
    Returns:
        None. The function modifies the peft_config object in place.
    
    Raises:
        ValueError: If 'num_layers', 'token_dim', or 'num_attention_heads' is not specified in peft_config or model_config.
    """
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
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = shifted_input_ids.masked_fill(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class Conv1D(nn.Module):
    """
    1D-convolutional layer Basically works like a linear layer but the weights are transposed.

    Args:
        n_out (`int`): The number of output features.
        n_in (`int`): The number of input features.
    """
    def __init__(self, n_out, n_in):
        r"""
        Initializes an instance of the Conv1D class.
        
        Args:
            self (Conv1D): The instance of the Conv1D class.
            n_out (int): The number of output channels or filters.
            n_in (int): The number of input channels or filters.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        """
        super().__init__()
        self.n_out = n_out
        self.weight = Parameter(initializer(Normal(sigma=0.02), (n_in, n_out), mindspore.float32))
        self.bias = Parameter(ops.zeros(n_out, mindspore.float32))

    def forward(self, x):
        r"""
        Constructs the output of a 1D convolutional layer.
        
        Args:
            self (Conv1D): The instance of the Conv1D class.
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_features) representing the input data.
        
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, n_out) representing the result of the 1D convolution operation.
        
        Raises:
            - ValueError: If the input tensor 'x' does not have the expected shape.
            - RuntimeError: If an error occurs during the matrix multiplication or bias addition operations.
        """
        size_out = x.shape[:-1] + (self.n_out,)
        x = self.matmul(x.view(-1, x.shape[-1]), self.weight) + self.bias
        x = x.view(size_out)
        return x
