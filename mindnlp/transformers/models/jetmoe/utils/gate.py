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
JetMoE Gate.
"""
from mindspore import nn, ops
from mindnlp.modules.functional import normalize

class top_k_gating(nn.Cell):

    """
    This class represents a top-k gating mechanism for selecting a subset of experts based on input logits. The gating mechanism uses the top-k experts to make predictions and can be used for model ensembling
or expert selection. The class inherits from nn.Cell and implements methods for initializing the mechanism, computing auxiliary loss, and constructing the top-k gating for input data. Additionally, the class
provides a method for returning an extra representation string for the module.
    
    The top_k_gating class provides the following methods:
    
    - __init__(input_size, num_experts, top_k): Initialize the top-k gating mechanism with specified input size, number of experts, and top-k value.
    - extra_repr(): Return extra representation string for the module, specifically the top-k value and number of experts.
    - compute_aux_loss(probs, logits, gates): Calculate and return the auxiliary loss based on the accumulated statistics.
    - construct(x): Compute the top-k gating for the input and return the top-k indices, top-k gating values, probability values for each expert, gates, and load.
    
    For more details, refer to the paper: https://arxiv.org/abs/1701.06538.
    """
    def __init__(
        self,
        input_size,
        num_experts,
        top_k,
    ):
        """
        Initialize the top-k gating mechanism.

        Args:
            input_size (int): Size of the input.
            num_experts (int): Number of experts.
            top_k (int): Number of top experts to select.
            acc_aux_loss (bool): Whether to accumulate auxiliary loss statistics.
            dropout (float): Dropout rate for gating network.
            hidden_size (int): Hidden size of the gating network.
            sample_topk (int): Number of top-k experts to sample during training.
            aux_loss (str): Type of auxiliary loss ('mi' or 'switch').
            gate_type (str): Type of gating mechanism ('mlp', 'linear', or 'gmm').
        """
        super().__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        assert top_k <= num_experts
        self.top_k = top_k

        self.layer = nn.Dense(input_size, num_experts, has_bias=False)

    def extra_repr(self):
        """
        Return extra representation string for the module.
        """
        return "k={}, num_experts={}".format(self.top_k, self.num_experts)

    def compute_aux_loss(self, probs, logits, gates):
        """
        Calculate and return the auxiliary loss based on the accumulated statistics.

        Args:
            eps (float): Small epsilon value for numerical stability.

        Returns:
            torch.Tensor: The calculated auxiliary loss.
        """
        count = logits.shape[0]
        probs = probs.sum(0)
        freq = (gates > 0).float().sum(0)
        lsesq = (ops.log(ops.exp(logits).sum(axis=-1)) ** 2).sum()

        switchloss = self.num_experts * (normalize(probs, p=1, dim=0) * normalize(freq, p=1, dim=0)).sum()
        zloss = lsesq / count
        loss = switchloss + 0.1 * zloss

        return loss

    def construct(self, x):
        """
        Compute the top-k gating for the input.

        See paper: https://arxiv.org/abs/1701.06538.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, input_size].
            skip_mask (torch.Tensor): Skip mask tensor (binary) with the same shape as `x`.
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float

        Returns:
            torch.Tensor: Top-k indices.
            torch.Tensor: Top-k gating values.
            torch.Tensor: Probability values for each expert.
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        logits = self.layer(x).float()
        top_k_logits, top_k_indices = ops.topk(logits, self.top_k, dim=1)
        top_k_gates = ops.softmax(top_k_logits, axis=1).type_as(x)

        if self.training:
            probs = ops.softmax(logits, axis=1)
            zeros = ops.zeros_like(probs)
            zeros = zeros.to(top_k_gates.dtype)  # Convert zeros to match top_k_gates dtype
            gates = zeros.scatter(1, top_k_indices, top_k_gates)
            self.loss = self.compute_aux_loss(probs, logits, gates)
        else:
            self.loss = 0

        return top_k_indices, top_k_gates
