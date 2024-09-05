# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager

import packaging.version
import mindnlp
from mindnlp import transformers
import mindspore
from mindnlp.core import nn
from mindnlp.core import ops

def dequantize_module_weight(module: nn.Module) -> nn.Parameter:
    """
    Helper function to dequantize a quantized weight.

    This function should be extended if more quantization schemes are added to the library.

    If the weight is not quantized, it will be returned as is.
    """
    if hasattr(module, "W_q"):  # For handling HQQ quantized weight
        weight = module.dequantize()
        return weight

    weight = module.weight
    # weight = nn.Parameter(weight)
    if not isinstance(weight, nn.Parameter):
        if isinstance(weight, mindspore.Tensor):
            # this is an FSDP-specific edge case
            return weight  # type: ignore
        raise TypeError(f"Input weight should be of type nn.Parameter, got {type(weight)} instead")

    cls_name = weight.__class__.__name__
    if cls_name not in ("Params4bit", "Int8Params"):
        return weight

    quant_state = getattr(module, "state", None)
    device = weight.device
    is_cpu = device.type == "cpu"
    weight = dequantize_bnb_weight(weight, state=quant_state)  # no-op if not bnb
    if is_cpu:
        # dequantize_bnb_weight for 8bit moves the device in-place, thus we need to move it back to CPU if necessary
        module.weight = module.weight.to(device)
    return weight


def dequantize_bnb_weight(weight: nn.Parameter, state=None):
    """Helper function to dequantize 4bit or 8bit bnb weights.

    Since dequantization is not supported on CPU, the weight will be temporarily moved to CUDA if necessary.
    """
    import bitsandbytes as bnb

    # BNB requires CUDA weights
    device = weight.device
    is_cpu = device.type == "cpu"
    if is_cpu:
        weight = weight.to("cuda")

    cls_name = weight.__class__.__name__
    if cls_name == "Params4bit":
        dequantized = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        if is_cpu:
            dequantized = dequantized.to(device)
        return dequantized

    if state.SCB is None:
        state.SCB = weight.SCB

    im = ops.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
    im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
    im, Sim = bnb.functional.transform(im, "col32")
    if state.CxB is None:
        state.CxB, state.SB = bnb.functional.transform(weight.data, to_order=state.formatB)
    out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
    dequantized = bnb.functional.mm_dequant(out32, Sout32, SCim, state.SCB, bias=None).t()
    if is_cpu:
        dequantized = dequantized.to(device)
    return dequantized