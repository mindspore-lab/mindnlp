"""
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""


from typing import Dict, List, Optional, Tuple, Union
from collections.abc import Mapping

import numpy as np
import mindspore as ms

from mindspore import ops
from mindspore import Tensor

#与huggingface.transformers同路程
from mindnlp.transformers.generation import TopKLogitsWarper, TopPLogitsWarper

#暂时只更改import_utils中的这两个函数
#from import_utils import is_npu_available, is_xpu_available

#如果遇到需要padding的时候，补充的数为-1
WANDB_PADDING = -1


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens 
            with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k,
                                  filter_value=filter_value,
                                  min_tokens_to_keep=min_tokens_to_keep)(None, logits)

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p,
                                  filter_value=filter_value,
                                  min_tokens_to_keep=min_tokens_to_keep)(None, logits)

    return logits


def flatten_dict(nested: Dict, sep: str = "/") -> Dict:
    """Flatten dictionary and concatenate nested keys with separator."""

    def recurse(nest: Dict, prefix: str, into: Dict) -> None:
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, Mapping):
                recurse(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    recurse(nested, "", flat)
    return flat


def convert_to_scalar(stats: Dict) -> Dict:
    """
    Converts the stats from a flattened dict to single scalar dicts
    """
    tensorboard_stats = {}
    for k, v in stats.items():
        # for tensorboard compatibility - arrays and tensors are ignored with tensorboard
        # therefore we convert single element tensors to scalars
        if isinstance(v, (Tensor,np.ndarray)) and (
            len(v.shape) == 0 or (len(v.shape) == 1 and v.shape[0] == 1)
        ):
            v = v.item()
        tensorboard_stats[k] = v
    return tensorboard_stats


def pad_sequence(sequences, padding_value=0):
    """
    Padding a set of sequences to make all sequences the same length
    """
    # Find the maximum length of the sequences
    max_len = max(seq.shape[0] for seq in sequences)
    padded_seqs = []
    for seq in sequences:
        # Calculate the padding needed
        pad_len = max_len - seq.shape[0]
        # Pad the sequence
        # pad_seq = ops.Pad(((0, pad_len),),)(seq)
        pad_seq = ops.pad(seq, (0, pad_len), mode='constant', value=padding_value)
        padded_seqs.append(pad_seq)
    return ops.stack(padded_seqs)

def stack_dicts(stats_dicts):
    """Stack the values of a dict."""
    results = {}
    for k in stats_dicts[0]:
        stats_list = [ops.reshape(d[k], (-1,)) for d in stats_dicts]
        results[k] = pad_sequence(stats_list, padding_value=WANDB_PADDING)
    return results


def add_suffix(input_dict: Dict, suffix: str) -> Dict:
    """Add suffix to dict keys."""
    return {k + suffix: v for k, v in input_dict.items()}


def pad_to_size(tensor: Tensor, size: int, dim: int = 1, padding: int = 50256) -> Tensor:
    """Pad tensor to size."""
    t_size = tensor.shape[dim]
    if t_size == size:
        return tensor
    return ops.pad(tensor, (0, size - t_size), "constant", padding)

def whiten(values: Tensor, shift_mean: bool = True) -> Tensor:
    """Whiten values."""
    mean, var = ops.mean(values), ops.var(values)
    whitened = (values - mean) * ops.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

def masked_mean(values: Tensor,
                mask: Tensor,
                axis: Optional[bool] = None) -> Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    return (values * mask).sum() / mask.sum()

def masked_var(values: Tensor,
               mask: Tensor,
               unbiased: bool = True,
               axis: Optional[bool] = None) -> Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask, axis = axis)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance

def masked_whiten(values: Tensor, mask: Tensor, shift_mean: bool = True) -> Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    #rsqrt逐元素计算输入Tensor元素的平方根倒数。
    whitened = (values - mean) * ops.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

def average_torch_dicts(list_of_dicts: List[Dict]) -> Dict:
    """Average values of a list of dicts with torch tensors."""
    average_dict = {}
    stack = ops.Stack()
    rmean = ops.ReduceMean()
    for key in list_of_dicts[0].keys():
        average_dict[key] = rmean(stack([d[key] for d in list_of_dicts]), axis=0)
    return average_dict

def entropy_from_logits(logits: Tensor) -> Tensor:
    """Calculate entropy from logits."""
    pd = ops.Softmax(axis=-1)(logits)
    entropy = ops.logsumexp(logits, -1) - ops.ReduceSum()(pd * logits, -1)
    return entropy

def clip_by_value(x: Tensor, tensor_min: float, tensor_max: float) -> Tensor:
    """
    Tensor extension to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = ops.maximum(
        ops.minimum(x, Tensor(tensor_max, dtype=x.dtype)),
        Tensor(tensor_min, dtype=x.dtype)
        )
    return clipped

def stats_to_np(stats_dict: Dict) -> Dict:
    """Cast all torch.tensors in dict to numpy arrays."""
    new_dict = {}
    for k, v in stats_dict.items():
        if isinstance(v, Tensor):
            new_dict[k] = v.numpy()
        elif isinstance(v, (int, float)):
            new_dict[k] = float(v)
        else:
            new_dict[k] = v
    return new_dict

def set_seed(seed: int) -> None:
    """
    Helper function for reproducible behavior to 
    set the seed in `random`, `numpy`, and `mindspore`.

    Args:
        seed (`int`): The seed to set.
    """
    np.random.seed(seed)
    ms.set_seed(seed)

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List[np.random.Generator],
                              np.random.Generator]] = None,
) -> Tensor:
    """A helper function to create random tensors
    on the desired `device` with the desired `dtype`. When
    passing a list of generators,
    you can seed each batch size individually.
    If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    batch_size = shape[0]

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            ops.standard_normal(shape, seed=generator[i])
            for i in range(batch_size)
        ]
        latents = ops.concat(latents, axis=0)
    else:
        latents = ops.standard_normal(shape, seed=generator)

    return latents

class LengthSampler:
    """
    Samples a length
    """

    def __init__(self, min_value: int, max_value: int):
        self.values = list(range(min_value, max_value))

    def __call__(self) -> int:
        return np.random.choice(self.values)




#######################################################

# def respond_to_batch(
#     model: nn.Module, queries: List[torch.LongTensor],
# txt_len: int = 20, top_k: int = 0, top_p: float = 1.0
# ) -> torch.LongTensor:
#     """Sample text from language model."""
#     input_ids = queries
#     for _i in range(txt_len):
#         # Get Logits
#         outputs = model(input_ids)
#         next_token_logits = outputs[0][:, -1, :]
#         next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
#         # Sample
#         probs = F.softmax(next_token_logits, dim=-1)
#         next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
#         input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
#     return input_ids[:, -txt_len:]


# class PPODecorators:
#     optimize_device_cache = False

#     @classmethod
#     @contextmanager
#     def empty_device_cache(cls):
#         yield
#         if cls.optimize_device_cache:
#             if is_xpu_available():
#                 gc.collect()
#                 torch.xpu.empty_cache()
#                 gc.collect()
#             elif is_npu_available():
#                 gc.collect()
#                 torch.npu.empty_cache()
#                 gc.collect()
#             elif torch.cuda.is_available():
#                 gc.collect()
#                 torch.cuda.empty_cache()
#                 gc.collect(







# Usage example:
# generator = np.random.default_rng(seed=42)
# randn_tensor((3, 4), generator=generator, device='GPU', dtype=mindspore.float32)
