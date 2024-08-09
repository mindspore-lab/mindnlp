'''
Module for chat format setup and model utilities.

This module provides functions for setting up chat format by adding special tokens to a tokenizer
and extending the embedding layer of a model based on the new special tokens. It also includes
utilities for removing and adding optimizer hooks from a DeepSpeed ZeRO-3 model, as well as a
context manager for unwrapping a model for generation.

Functions:
- setup_chat_format: Set up chat format by modifying tokenizer and model.
- remove_hooks: Remove optimizer hooks from a DeepSpeed ZeRO-3 model.
- add_hooks: Add optimizer hooks to a DeepSpeed ZeRO-3 model.
- unwrap_model_for_generation: Context manager to unwrap a model for generation.

Classes:
- ChatMlSpecialTokens: Dataclass for special tokens used in ChatML.

Constants:
- SUPPORTED_ARCHITECTURES: Tuple of supported model architectures.

'''

# from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from mindnlp.transformers import PreTrainedModel, PreTrainedTokenizer

from .modeling_value_head import AutoModelForCausalLMWithValueHead
from .modeling_value_head import AutoModelForSeq2SeqLMWithValueHead

# from .modeling_base import PreTrainedModelWrapper

SUPPORTED_ARCHITECTURES = (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
)



# if is_deepspeed_available():
#     import deepspeed

# if TYPE_CHECKING:
#     from accelerate import Accelerator
#     from deepspeed.runtime.engine import DeepSpeedEngine
#     from torch.nn.parallel.distributed import DistributedDataParallel


'''(TO)(DO): Add Abstract Base Class if more formats are added'''
@dataclass
class ChatMlSpecialTokens:
    """Dataclass for special tokens used in ChatML,
    including system, user, assistant, bos, eos, and pad tokens."""

    bos_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"
    pad_token: str = "<|im_end|>"

    @property
    def system(self):
        '''return system'''
        return f"{self.bos_token}system"

    @property
    def user(self):
        '''return user'''
        return f"{self.bos_token}user"

    @property
    def assistant(self):
        '''return assistant'''
        return f"{self.bos_token}assistant"

    @property
    def chat_template(self):
        '''you can check your chat_template here'''
        return (
            "{% for message in messages %}"
            f"{{{{'{self.bos_token}' + message['role'] + '\n' \
                + message['content'] + '{self.eos_token}' + '\n'}}}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            f"{{{{ '{self.assistant}\n' }}}}"
            "{% endif %}"
        )


FORMAT_MAPPING = {"chatml": ChatMlSpecialTokens}


def setup_chat_format(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    formatt: Optional[Literal["chatml"]] = "chatml",
    resize_to_multiple_of: Optional[int] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Setup chat format by adding special tokens to the tokenizer, setting the correct format,
    and extending the embedding layer of the model based on the new special tokens.

    Args:
      model (`~transformers.PreTrainedModel`): The model to be modified.
      tokenizer (`~transformers.PreTrainedTokenizer`): The tokenizer to be modified.
      format (`Optional[Literal["chatml"]]`): The format to be set. Defaults to "chatml".
      resize_to_multiple_of (`Optional[int]`): Number to resize the embedding layer to.
      Defaults to None.
    Returns:
      model (`~transformers.PreTrainedModel`): The modified model.
      tokenizer (`~transformers.PreTrainedTokenizer`): The modified tokenizer.
    """
    # check if format available and retrieve
    if formatt not in FORMAT_MAPPING:
        raise ValueError(f"Format {formatt} not available. \
            Please use one of {FORMAT_MAPPING.keys()}")

    chat_format = FORMAT_MAPPING[formatt]()

    # set special tokens and them
    tokenizer.eos_token = chat_format.eos_token
    tokenizer.pad_token = chat_format.pad_token
    tokenizer.bos_token = chat_format.bos_token
    tokenizer.add_special_tokens({"additional_special_tokens": \
        [chat_format.bos_token, chat_format.eos_token]})
    # set chat format for tokenizer
    tokenizer.chat_template = chat_format.chat_template

    # resize embedding layer to a multiple of 64, https://x.com/karpathy/status/1621578354024677377
    model.resize_token_embeddings(
        len(tokenizer), pad_to_multiple_of=resize_to_multiple_of \
            if resize_to_multiple_of is not None else None
    )
    # Update the model config to use the new eos & bos tokens
    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
    # Update the generation config to use the new eos & bos token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def remove_hooks(model: "DeepSpeedEngine") -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    optimizer_offload = None

    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []


def add_hooks(model: "DeepSpeedEngine") -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    optimizer_offload = None
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer

    optimizer_offload._register_hooks_recursively(optimizer_offload.module)


# @contextmanager
# def unwrap_model_for_generation(
#     model: Union["DistributedDataParallel", "DeepSpeedEngine"],\
#         accelerator: "Accelerator", is_peft_model: bool = False
# ) -> Union["PreTrainedModelWrapper", "DeepSpeedEngine"]:
#     """Context manager to unwrap a model for generation.

#     For ZeRO-3 models, we gather the weights once to speed up generation.
#     """
#     unwrapped_model = accelerator.unwrap_model(model)
#     if is_peft_model:
#         unwrapped_model.pretrained_model.disable_adapter()
#     if accelerator.state.deepspeed_plugin is not None\
#         and accelerator.state.deepspeed_plugin.zero_stage == 3:
#         with deepspeed.zero.GatheredParameters(model.parameters()):
#             remove_hooks(model)
#             yield model
#             add_hooks(model)
#     else:
#         yield unwrapped_model
