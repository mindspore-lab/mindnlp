# coding=utf-8
# Copyright 2024 BlinkDL, et al.
# Copyright 2024 yuunnn-w, et al.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications copyright 2024 [Huawei Technologies Co., Ltd]
# Changes: Migrated to MindSpore interface
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import mindspore
from mindspore import ops

def old_sample_logits(out: mindspore.Tensor, temperature: float = 1.0, top_p: float = 0.8) -> mindspore.Tensor:
    """
    对模型输出的logits进行采样。

    Args:
        out (ops.Tensor): 模型输出的logits张量,形状为[Batch, vocab_size]。
        temperature (float): 温度参数,用于调节采样的多样性,默认为1.0。
        top_p (float): Top-p截断参数,用于稳定和控制采样概率分布,默认为0.8。

    Returns:
        ops.Tensor: 采样结果,形状为[Batch, 1],每个元素表示一个样本中采样得到的词的索引。
    """
    # 确保top_p和temperature都是非负值
    top_p = max(0.0, min(1.0, top_p))
    temperature = max(0.0, temperature)

    # 将out转换为概率分布
    probs = ops.softmax(out, dim=-1)

    # 根据top_p截断概率分布
    sorted_probs, _ = ops.sort(probs, descending=True)
    cumulative_probs = ops.cumsum(sorted_probs, dim=-1)
    cutoff_mask = (cumulative_probs > top_p).float()
    cutoff_index = ops.argmax(cutoff_mask * ops.arange(cutoff_mask.shape[-1], device=cutoff_mask.device).float(), dim=-1)
    cutoff_values = sorted_probs.gather(-1, cutoff_index.unsqueeze(-1)).squeeze(-1)
    probs = ops.where(probs < cutoff_values.unsqueeze(-1), ops.zeros_like(probs), probs)

    # 对概率分布进行温度调节
    if temperature != 1.0:
        probs = ops.pow(probs, 1.0 / temperature)

    # 归一化概率分布
    probs /= ops.sum(probs, dim=-1, keepdim=True)

    # 如果top_p为0,则选择概率最大的位置;否则按照概率分布随机采样
    if top_p != 0:
        sampled_indices = ops.multinomial(probs, num_samples=1)
    else:
        sampled_indices = ops.argmax(probs, dim=-1, keepdim=True)
        

    return sampled_indices

def sample_logits(out: ops.Tensor, temperature: float = 1.0, top_p: float = 0.8) -> ops.Tensor:
    """
    Sample from the logits tensor produced by the model.

    Args:
        out (ops.Tensor): Logits tensor from the model, shape [* , vocab_size].
        temperature (float): Temperature parameter for controlling the diversity of sampling. Default is 1.0.
        top_p (float): Top-p truncation parameter for stabilizing and controlling the sampling probability distribution. Default is 0.8.

    Returns:
        ops.Tensor: Sampled indices, shape [*].
    """
    # Apply temperature scaling
    scaled_logits = out / temperature

    # Convert logits to probabilities
    probabilities = ops.softmax(scaled_logits, axis=-1)

    # Sort the probabilities to identify the top-p candidates
    sorted_probs, sorted_indices = ops.sort(probabilities, descending=True)

    # Compute the cumulative distribution of probabilities
    cumulative_probs = ops.cumsum(sorted_probs, axis=-1)

    # Remove tokens with a cumulative probability above the threshold (top_p)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].copy()
    sorted_indices_to_remove[..., 0] = 0

    # Create a mask for the indices to remove
    indices_to_remove = sorted_indices_to_remove.scatter(axis=-1, index=sorted_indices, src=sorted_indices_to_remove)

    # Use the mask to zero out probabilities that should be removed
    probabilities[indices_to_remove] = 0.0

    # Resample if probabilities are all zero (unlikely but just in case)
    if ops.all(probabilities == 0):
        probabilities = ops.ones_like(probabilities)
        probabilities /= probabilities.sum()

    # Sample from the modified distribution
    sampled_indices = ops.multinomial(probabilities, 1)

    return sampled_indices.squeeze(-1)