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
JetMoE Experts.
"""
import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import Uniform, initializer

def compute_gating(k: int, num_experts: int, top_k_gates: mindspore.Tensor, top_k_indices: mindspore.Tensor):
    """
    Compute gating values for the mixture of experts based on probabilities and top-k indices.

    Args:
        k (int): Number of experts to select.
        num_experts (int): Total number of experts.
        top_k_gates (mindspore.Tensor): Gating values for top-k experts (batch_size x k).
        top_k_indices (mindspore.Tensor): Indices of top-k experts (batch_size x k).

    Returns:
        mindspore.Tensor: Batch-level gating values.
        mindspore.Tensor: Batch-level expert indices.
        mindspore.Tensor: Expert size for each expert.
        mindspore.Tensor: Sorted indices of top-k experts.
    """
    zeros = ops.zeros([top_k_gates.shape[0], num_experts], dtype=top_k_gates.dtype)
    gates = zeros.scatter(1, top_k_indices, ops.ones(top_k_indices.shape, dtype=top_k_gates.dtype))
    expert_size = gates.long().sum(0)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    _, index_sorted_experts = top_k_experts.sort(0)
    batch_index = index_sorted_experts.div(k, rounding_mode="trunc")
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size, index_sorted_experts


class ParallelExperts(nn.Cell):

    """
    Represents a module for parallel experts within a neural network architecture. 
    
    This class inherits from nn.Cell and provides functionality for parallel processing of inputs by multiple experts. 
    The ParallelExperts module initializes with the specified number of experts, input size, and output size. 
    It includes methods for the forward pass operation to process input tensors through the experts and generate an output tensor.
    
    Methods:
    - __init__(self, num_experts, input_size, output_size): Initializes the ParallelExperts module with the given parameters.
    - extra_repr(self): Returns a string representation of the module with details on num_experts, input_size, and output_size.
    - construct(self, inputs, expert_size): Performs the forward pass operation by splitting input tensors among experts,
      applying operations for each expert using the weight parameters, and concatenating the outputs to form the final result tensor.
    """
    def __init__(self, num_experts, input_size, output_size) -> None:
        """
        Initialize the ParallelExperts module.

        Args:
            num_experts (int): Number of experts.
            input_size (int): Size of the input.
            output_size (int): Size of the output.
            bias (bool): Whether to include bias terms.
        """
        super().__init__()
        self.weight = Parameter(initializer(Uniform(1.0 / output_size), (num_experts, output_size, input_size)))
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def extra_repr(self):
        """
        Method 'extra_repr' in the class 'ParallelExperts' generates a string representation of the object for debugging and logging purposes.
        
        Args:
            self: An instance of the 'ParallelExperts' class.
                Type: Object
                Purpose: Represents the current instance of the class.
                Restrictions: None
        
        Returns:
            A formatted string containing information about the number of experts, input size, and output size of the object.
                Type: None
                Purpose: Returns None as the formatted string is directly printed or used for logging purposes.
        
        Raises:
            None
        """
        return "num_experts={}, input_size={}, output_size={}".format(
            self.num_experts, self.input_size, self.output_size
        )

    def construct(self, inputs, expert_size):
        """
        Forward pass of the ParallelExperts module.

        Args:
            inputs (Tensor): Input tensor.
            expert_size: Expert size information.

        Returns:
            Tensor: Output tensor.
        """
        input_list = inputs.split(expert_size, axis=0)
        output_list = []
        for i in range(self.num_experts):
            output_list.append(ops.dense(input_list[i], self.weight[i]))
        results = ops.cat(output_list, axis=0)
        return results
