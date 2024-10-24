# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""trl trainer utils, only contents dpo related class and func."""
import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindnlp.core import no_grad
from ..core import pad_sequence


def pad(
    tensors: List[mindspore.Tensor],
    padding_value: int = 0,
    padding_side: str = "right"
) -> mindspore.Tensor:
    """
    Pad input function.
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = mindspore.ops.full(
        (len(tensors), *output_shape),
        padding_value,
        dtype=tensors[0].dtype
        )
    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.

    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def encoder_decoder_pad(self, features, k):
        """When model is encoder-decoder arch, do pad."""
        to_pad = [mindspore.Tensor(ex[k]) for ex in features]
        if (k.startswith("prompt")) and (k.endswith("input_ids")):
            if self.pad_token_id is None:
                raise ValueError(
                    "Padding is enabled, but the tokenizer is "
                    " not configured with a padding token."
                    " Explicitly set `tokenizer.pad_token` "
                    "(e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                    " before calling the trainer."
                )
            padding_value = self.pad_token_id
        elif k.endswith("_attention_mask"):
            padding_value = 0
        elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
            padding_value = self.label_pad_token_id
        else:
            raise ValueError(f"Unexpected key in batch '{k}'")
        return to_pad, padding_value

    def non_encoder_decoder_pad(self, features, k):
        """when model is not encoder-decoder arch, do pad."""
        # Set padding value based on the key
        if k.endswith("_input_ids"):
            if self.pad_token_id is None:
                raise ValueError(
                    "Padding is enabled, but the tokenizer is "
                    "not configured with a padding token."
                    " Explicitly set `tokenizer.pad_token` "
                    "(e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                    " before calling the trainer."
                )
            padding_value = self.pad_token_id
        elif k.endswith("_labels"):
            padding_value = self.label_pad_token_id
        elif k.endswith("_attention_mask"):
            padding_value = 0
        elif k.endswith("_pixel_values"):
            padding_value = 0
        else:
            raise ValueError(f"Unexpected key in batch '{k}'")

        # Set padding side based on the key
        if k in ["prompt_input_ids", "prompt_attention_mask"]:
            padding_side = "left"
        else:
            padding_side = "right"

        # Set the dtype
        if k.endswith("_pixel_values"):
            dtype = mindspore.float32  # will be downcasted if necessary by the Trainer
        else:
            dtype = mindspore.int64
        # Convert to tensor and pad
        to_pad = [mindspore.tensor(ex[k], dtype=dtype).squeeze(0) for ex in features]
        return to_pad, padding_value, padding_side

    def __call__(self, features):
        # first, pad everything to the same length
        padded_batch = {}

        for k in features[0].keys():
            if k.endswith(("_input_ids", "_attention_mask", "_labels", "_pixel_values")):
                if self.is_encoder_decoder:
                    to_pad, padding_value = self.encoder_decoder_pad(features, k)
                    padded_batch[k] = pad_sequence(
                        to_pad,
                        padding_value=padding_value
                    )
                else:
                    to_pad, padding_value, padding_side = self.non_encoder_decoder_pad(
                        features, k
                    )
                    padded_batch[k] = pad(
                        to_pad,
                        padding_value=padding_value,
                        padding_side=padding_side
                    )
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = mindspore.tensor(features[k], dtype=mindspore.float32)
            else:
                padded_batch[k] = [ex[k] for ex in features]
        return padded_batch


@dataclass
class RunningMoments:
    """
    Calculates the running mean and standard deviation of a data stream. Reference:
    https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L75
    """
    mean: float = 0
    std: float = 1
    var: float = 1
    count: float = 1e-24

    @no_grad()
    def update(self, xs: mindspore.Tensor) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        # if self.accelerator.use_distributed:
            # xs_mean, xs_var, xs_count = get_global_statistics(self.accelerator, xs)
        # else:
        xs_count = xs.numel()
        xs_var, xs_mean = mindspore.ops.var_mean(xs)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += (delta * xs_count / tot_count).item()
        new_var = tot_sum / tot_count
        self.std = (new_var * tot_count / (tot_count - 1)).float().sqrt().item()
        self.var = new_var.item()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        # save everything except accelerator
        save_dict = dataclasses.asdict(
            self,
            dict_factory=lambda x: {k: v for (k, v) in x if k != "accelerator"}
        )
        json_string = json.dumps(save_dict, indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, accelerator, json_path: str):
        """Create an instance from the content of `json_path`."""
        # load everything except accelerator
        with open(json_path, encoding="utf-8") as f:
            text = f.read()
        return cls(accelerator=accelerator, **json.loads(text))


def add_bos_token_if_needed(
    bos_token_id: Optional[int],
    prompt_len_input_ids: int,
    prompt_tokens: Dict[str, List[int]],
    chosen_prompt_len_input_ids: int,
    chosen_tokens: Dict[str, List[int]],
    rejected_prompt_len_input_ids: int,
    rejected_tokens: Dict[str, List[int]],
):
    """
    Add BOS token if needed.
    """
    if bos_token_id is not None:
        if prompt_len_input_ids == 0 or \
            bos_token_id != prompt_tokens["prompt_input_ids"][0]:
            prompt_tokens["prompt_input_ids"] = (
                [bos_token_id] + prompt_tokens["prompt_input_ids"]
            )
            prompt_tokens["prompt_attention_mask"] = (
                [1] + prompt_tokens["prompt_attention_mask"]
            )
        if chosen_prompt_len_input_ids == 0 or \
            bos_token_id != chosen_tokens["prompt_input_ids"][0]:
            chosen_tokens["prompt_input_ids"] = (
                [bos_token_id] + chosen_tokens["prompt_input_ids"]
            )
            chosen_tokens["prompt_attention_mask"] = (
                [1] + chosen_tokens["prompt_attention_mask"]
            )
        if rejected_prompt_len_input_ids == 0 or \
            bos_token_id != rejected_tokens["prompt_input_ids"][0]:
            rejected_tokens["prompt_input_ids"] = (
                [bos_token_id] + rejected_tokens["prompt_input_ids"]
            )
            rejected_tokens["prompt_attention_mask"] = (
                [1] + rejected_tokens["prompt_attention_mask"]
            )
    return prompt_tokens, chosen_tokens, rejected_tokens


def add_eos_token_if_needed(
    eos_token_id: int,
    chosen_tokens: Dict[str, List[int]],
    rejected_tokens: Dict[str, List[int]]
):
    """
    Add EOS Token if needed.
    """
    if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
        chosen_tokens["input_ids"].append(eos_token_id)
        chosen_tokens["attention_mask"].append(1)
    if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
        rejected_tokens["input_ids"].append(eos_token_id)
        rejected_tokens["attention_mask"].append(1)
    return chosen_tokens, rejected_tokens


def get_exp_cap(value, decimal=4):
    """
    Get the exponent cap of a value. This is used to cap the exponent of a value to avoid overflow.
    The formula is : log(value.dtype.max)
    E.g.
      For float32 data type, the maximum exponent value is 88.7228 to 4 decimal points.
    ```

    Args:
        value (`torch.Tensor`):
            The input tensor to obtain the data type
        decimal (`int`):
            The number of decimal points of the output exponent cap.
            eg: direct calling exp(log(torch.float32.max)) will result in inf
            so we cap the exponent to 88.7228 to avoid overflow.
    """
    vdtype_max = mindspore.ops.zeros([1]).to(value.dtype)\
         + mindspore.tensor(mindspore.dtype_to_nptype(np.finfo(value.dtype).max))
    vdtype_log_max = mindspore.ops.log(vdtype_max).to(value.device)
    if decimal > 0:
        return mindspore.ops.floor(vdtype_log_max * 10**decimal) / 10**decimal
    return vdtype_log_max


def cap_exp(value, cap=-1):
    """
    Cap the exponent value below the upper-bound to avoid overflow,
    before calling torch.exp
    """
    cap = get_exp_cap(value) if cap < 0 else cap
    return mindspore.ops.exp(mindspore.ops.clamp(value, max=cap))


def disable_dropout_in_model(model: mindspore.nn.Cell) -> None:
    """
    disable all dropout layer in given model.
    """
    for module in model.modules():
        if isinstance(module, mindspore.nn.Dropout):
            module.p = 0

def pad_to_length(
        tensor: mindspore.Tensor,
        length: int,
        pad_value: Union[int, float],
        dim: int = -1
    ) -> mindspore.Tensor:
    """
    pad all input tensor to the batch's maximal length.
    """
    if tensor.shape[dim] >= length:
        return tensor
    pad_size = list(tensor.shape)
    pad_size[dim] = length - tensor.shape[dim]
    return mindspore.ops.cat(
        [
            tensor,
            pad_value * mindspore.ops.ones(tuple(pad_size), dtype=tensor.dtype),
        ],
        axis=dim,
    )


def peft_module_casting_to_bf16(model):
    """
    Make peft model modules into the same precision.
    """
    for name, module in model.named_modules():
        if isinstance(module, mindspore.nn.LayerNorm) or "norm" in name:
            module = module.to(mindspore.float32)
        elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == mindspore.float32:
                    module = module.to(mindspore.bfloat16)
