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
JetMoE MoE.
"""
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer

from .parallel_experts import ParallelExperts, compute_gating

from .gate import top_k_gating


class MoE(nn.Cell):
    """
    A Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.

    Args:
        input_size: integer - size of the input
        hidden_size: integer - size of the expert's hidden layer
        num_experts: an integer - number of experts
        top_k: an integer - how many experts to use for each batch element
        bias: a boolean - whether to include bias in linear layers
        activation: an activation function to apply to expert's outputs
        glu: an boolean - whether to use GLU activation
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        num_experts,
        top_k,
        bias=True,
        activation=None,
        glu=True,
    ):
        """
        Initializes the MoE (Mixture of Experts) model with the specified parameters.
        
        Args:
            input_size (int): The size of the input feature vector.
            hidden_size (int): The size of the hidden layer.
            num_experts (int): The number of experts in the model.
            top_k (int): The top-k value used for expert selection.
            bias (bool): Indicates whether to include bias in the model.
            activation (str): The activation function to be applied. Default is None.
            glu (bool): Indicates whether to use Gated Linear Units (GLU) in the model.
        
        Returns:
            None. This method initializes the MoE model with the specified parameters.
        
        Raises:
            None.
        """
        super(MoE, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.glu = glu
        if bias:
            self.bias = Parameter(initializer('zeros', (input_size,)))
        else:
            self.bias = None

        self.input_linear = ParallelExperts(num_experts, input_size, hidden_size * 2 if glu else hidden_size)
        self.output_linear = ParallelExperts(num_experts, hidden_size, input_size)

        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

        self.router = top_k_gating(
            input_size=input_size,
            num_experts=num_experts,
            top_k=top_k,
        )

    def extra_repr(self):
        """
        This method generates a string representation of the MoE (Mixture of Experts) class instance.
        
        Args:
            self: MoE class instance. Represents the current instance of the MoE class.
        
        Returns:
            None. The method does not return any value directly but generates a formatted string representation 
            containing the top_k and num_experts attributes of the MoE instance.
        
        Raises:
            No exceptions raised by this method.
        """
        return "k={}, e={}".format(self.top_k, self.num_experts)

    def get_aux_loss_and_clear(self):
        """
        Get the accumulated auxiliary loss and clear it.

        Returns:
            float: Accumulated auxiliary loss.
        """
        return self.gate.get_aux_loss_and_clear()

    def compute_gate(self, x):
        """
        Compute the gate for selecting the expert to route the input data to.
        
        Args:
            self (MoE): An instance of the MoE class.
            x: The input data to be routed to an expert. It should be a tensor of shape (batch_size, input_size).
        
        Returns:
            None
        
        Raises:
            None
        
        This method computes the gate for selecting the expert to route the input data to. It follows the following steps:
        1. Calls the 'router' function to obtain the top-k indices and top-k gates.
        2. Calls the 'compute_gating' function to compute the batch-level gates, batch index, expert size, and sorted expert indices.
        3. Converts the expert size to a list.
        4. Returns the loss obtained from the 'router' function.
        
        Note: The 'router' and 'compute_gating' functions are assumed to be defined elsewhere in the code.
        """
        top_k_indices, self.top_k_gates = self.router(x)

        self.batch_gates, self.batch_index, expert_size, self.index_sorted_experts = compute_gating(
            self.top_k, self.num_experts, self.top_k_gates, top_k_indices
        )
        self.expert_size = expert_size.tolist()

        return self.router.loss

    def batch_forward(self, x):
        """
        Forward pass of the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.
            skip_mask (Tensor): Skip mask tensor.
            sample_topk (int): Number of experts to sample during training.
            multiply_by_gates (bool): Whether to multiply outputs by gating values.

        Returns:
            Tensor: Output tensor.
            float: Gating loss.
        """
        bsz, length, emb_size = x.shape
        x = x.reshape(-1, emb_size)
        loss = self.compute_gate(x)

        expert_inputs = x[self.batch_index]
        h = self.input_linear(expert_inputs, self.expert_size)
        if self.glu:
            h, g = h.chunk(2, axis=-1)
            h = self.activation(h) * g
        else:
            h = self.activation(h)
        expert_outputs = self.output_linear(h, self.expert_size)

        expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = ops.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        if self.bias is not None:
            y = y + self.bias
        return y, loss

    def single_forward(self, x):
        """
        This method performs a single forward pass through the Mixture of Experts (MoE) layer.
        
        Args:
            self (object): The instance of the MoE class.
                This parameter is a reference to the current instance of the MoE class.
                It is used to access the attributes and methods of the class.
            x (tensor): Input tensor of shape (batch_size, sequence_length, embedding_size).
                This tensor represents the input data to be processed by the MoE layer.
                The batch_size denotes the number of samples in a batch.
                The sequence_length denotes the length of each input sequence.
                The embedding_size represents the dimension of the input embeddings.
        
        Returns:
            None
            This method does not return any value but updates the internal state of the MoE layer.
        
        Raises:
            AttributeError: If the router or activation functions are not properly defined.
            ValueError: If the input tensor shape is not compatible with the expected dimensions.
        """
        bsz, length, emb_size = x.shape

        x = x.reshape(1, self.input_size)
        top_k_indices, top_k_gates = self.router(x)
        loss = self.router.loss

        y_list = []
        for i in range(self.top_k):
            expert_idx = top_k_indices[0, i]

            h = ops.dense(x, self.input_linear.weight[expert_idx])
            if self.glu:
                h, g = h.chunk(2, axis=-1)
                h = self.activation(h) * g
            else:
                h = self.activation(h)
            y = ops.dense(h, self.output_linear.weight[expert_idx]) * top_k_gates[0, i]

            y_list.append(y)

        y = sum(y_list)
        y = y.view(bsz, length, self.input_size)
        if self.bias is not None:
            y = y + self.bias
        return y, loss

    def construct(self, x):
        """
        Forward pass of the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        bsz, length, emb_size = x.shape
        if bsz * length == 1:
            return self.single_forward(x)
        else:
            return self.batch_forward(x)

    def single_map(self, x):
        """
        This method 'single_map' is a part of the 'MoE' class and is used to perform a single mapping operation in the Mixture of Experts (MoE) model.
        
        Args:
            self (object): The instance of the MoE class.
            x (tensor): The input tensor representing the input data for the single mapping operation. It should have a shape of (bsz, length, emb_size), where 'bsz' is the batch size, 'length' is the length
of input sequence, and 'emb_size' is the embedding size.
        
        Returns:
            y (tensor): The output tensor representing the result of the single mapping operation. It has a shape of (bsz, length, self.top_k, -1), where 'bsz' is the batch size, 'length' is the length of
input sequence, and 'self.top_k' is the number of top-k expert indices, and '-1' represents the output dimension.
            loss (float): The loss value associated with the single mapping operation.
        
        Raises:
            This method does not explicitly raise any exceptions.
        """
        bsz, length, emb_size = x.shape

        x = x.reshape(1, self.input_size)
        self.top_k_indices, self.top_k_gates = self.router(x)
        loss = self.router.loss

        y_list = []
        for i in range(self.top_k):
            expert_idx = self.top_k_indices[0, i]
            y = ops.dense(x, self.input_linear.weight[expert_idx])
            y_list.append(y)
        y = ops.cat(y_list, axis=0)
        y = y.view(bsz, length, self.top_k, -1)
        return y, loss

    def batch_map(self, x):
        """

        Args:
            x: tensor shape [batch_size, input_size]
            train: a boolean scalar.
            loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
            y: a tensor with shape [batch_size, output_size].
            extra_training_loss: a scalar.  This should be added into the overall
            training loss of the model.  The backpropagation of this loss
            encourages all experts to be approximately equally used across a batch.
        """
        """
        Map input through the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.
            skip_mask (Tensor): Skip mask tensor.
            sample_topk (int): Number of experts to sample during training.
            return_indices (bool): Whether to return expert indices.

        Returns:
            Tensor: Output tensor.
            float: Gating loss.
        """
        bsz, length, emb_size = x.shape
        x = x.reshape(-1, emb_size)
        loss = self.compute_gate(x)

        expert_inputs = x[self.batch_index]
        expert_outputs = self.input_linear(expert_inputs, self.expert_size)

        zeros = ops.zeros(
            (bsz * length * self.top_k, self.hidden_size), dtype=expert_outputs.dtype
        )
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.top_k, -1)
        return y, loss

    def map(self, x):
        """
        Map input through the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        bsz, length, emb_size = x.shape
        if bsz * length == 1:
            return self.single_map(x)
        else:
            return self.batch_map(x)

    def single_reduce(self, x):
        """
        Reduces the input tensor 'x' using a single mixture of experts (MoE) layer.
        
        Args:
            self (MoE): An instance of the MoE class.
            x (torch.Tensor): The input tensor of shape (batch_size, length, k, emb_size), where
                              - batch_size: The number of sequences in a batch.
                              - length: The length of each sequence.
                              - k: The number of experts.
                              - emb_size: The size of the embedding vector.
                              The input tensor represents the embeddings for each sequence.
        
        Returns:
            None
        
        Raises:
            None
        """
        bsz, length, k, emb_size = x.shape

        x = x.reshape(k, emb_size)

        y_list = []
        for i in range(self.top_k):
            expert_idx = self.top_k_indices[0, i]
            y = ops.dense(x[i].reshape(1, -1), self.output_linear.weight[expert_idx]) * self.top_k_gates[0, i]
            y_list.append(y)
        y = sum(y_list)
        y = y.view(bsz, length, self.input_size)
        if self.bias is not None:
            y = y + self.bias
        return y

    def batch_reduce(self, x):
        """
        Reduce the mapped output.

        Args:
            x (Tensor): Mapped output tensor.
            multiply_by_gates (bool): Whether to multiply outputs by gating values.

        Returns:
            Tensor: Reduced output tensor.
        """
        bsz, length, k, emb_size = x.shape
        x = x.reshape(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_linear(expert_inputs, self.expert_size)

        expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = ops.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        if self.bias is not None:
            y = y + self.bias
        return y

    def reduce(self, x):
        """
        Reduce the mapped output.

        Args:
            x (Tensor): Mapped output tensor.

        Returns:
            Tensor: Reduced output tensor.
        """
        bsz, length, k, emb_size = x.shape
        if bsz * length == 1:
            return self.single_reduce(x)
        else:
            return self.batch_reduce(x)
