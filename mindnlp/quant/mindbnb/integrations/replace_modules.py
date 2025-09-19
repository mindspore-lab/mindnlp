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
"""
    replace modules
"""
# pylint: disable=E0611, E0401
import logging

from bitsandbytes.nn.modules import Int8Params
import bitsandbytes as bnb

from mindtorch import nn
from mindtorch.nn import Parameter

logger = logging.getLogger(__name__)


def _replace_with_bnb_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
    llm_int8_has_fp16_weight=False,
    llm_int8_threshold=6.0,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if (
            isinstance(module, (nn.Conv1d, nn.Linear))
            and name not in modules_to_not_convert
        ):
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str)
                for key in modules_to_not_convert
            ):
                # Initialize empty weights if necessary (replace with MindSpore equivalent if required)
                if isinstance(module, nn.Conv1d):
                    in_features, out_features = module.weight.shape
                else:
                    in_features = module.in_features
                    out_features = module.out_features

                weight = model._modules[name].weight.clone()
                bias = (
                    model._modules[name].bias.clone()
                    if model._modules[name].bias is not None
                    else None
                )

                # Replace with MindSpore equivalent or custom module
                model._modules[name] = bnb.nn.Linear8bitLt(
                    in_features,
                    out_features,
                    has_fp16_weights=llm_int8_has_fp16_weight,
                    threshold=llm_int8_threshold,
                )

                model._modules[name].weight = Int8Params(
                    weight.data,
                    requires_grad=llm_int8_has_fp16_weight,
                    has_fp16_weights=llm_int8_has_fp16_weight,
                )
                if bias is not None:
                    model._modules[name].bias = Parameter(bias)

                model._modules[name].quant()
                has_been_replaced = True

                # Store the module class in case we need to transpose the weight later
                model._modules[name].source_cls = type(module)
                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad = False

        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_bnb_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)

    return model, has_been_replaced


def replace_with_bnb_linear(
    model, modules_to_not_convert=None, current_key_name=None, quantization_config=None
):
    modules_to_not_convert = (
        ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    )
    model, has_been_replaced = _replace_with_bnb_linear(
        model, modules_to_not_convert, current_key_name, quantization_config
    )

    if not has_been_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
