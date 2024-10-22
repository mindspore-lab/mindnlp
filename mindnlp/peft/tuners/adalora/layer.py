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
# ============================================================================
"Adalora Layer"
import warnings
from typing import Any, List, Optional

from mindspore import Tensor, get_grad

from mindnlp.core import nn, ops
from mindnlp.core.nn import Parameter
from mindnlp.core.nn import ParameterDict, ModuleDict
from mindnlp.peft.utils import transpose
from mindnlp.transformers.ms_utils import Conv1D

from ..tuners_utils import check_adapters_to_merge, BaseTunerLayer


class AdaLoraLayer(BaseTunerLayer):
    "AdaLoraLayer class for AdaLoraModel."
    # List all names of layers that may contain adapter weights
    # Note: ranknum doesn't need to be included as it is not an nn.Module
    adapter_layer_names = ("lora_A", "lora_B", "lora_E", "lora_embedding_A", "lora_embedding_B")
    # other_param_names is defined in LoraLayer

    def __init__(self, base_layer: nn.Module) -> None:
        r"""
        Initializes an instance of the AdaLoraLayer class.
        
        Args:
            self: The instance of the AdaLoraLayer class.
            base_layer (nn.Module): The base layer to be used for the AdaLoraLayer. It can be a Dense, Conv2d, Embedding, or Conv1D layer.
                                  For Dense and Conv2d layers, it extracts the input and output channel dimensions.
                                  For Embedding layers, it extracts the vocabulary size and embedding size.
                                  For Conv1D layers, it extracts the weight shape if available, otherwise the weight shape.
                                  Any other layer type will raise a ValueError.
                                  The base_layer is used to initialize the in_features and out_features attributes of the AdaLoraLayer.
        
        Returns:
            None
        
        Raises:
            ValueError: If the base_layer is not one of the supported layer types.
        """
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = ModuleDict()
        self.lora_E = ParameterDict({})
        self.lora_A = ParameterDict({})
        self.lora_B = ParameterDict({})
        self.ranknum = ParameterDict({})
        # For Embedding layer
        self.lora_embedding_A = ModuleDict()
        self.lora_embedding_B = ModuleDict()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        r"""
        This method updates the AdaLoraLayer with the provided parameters.
        
        Args:
            self (object): The instance of the AdaLoraLayer class.
            adapter_name (str): The name of the adapter to be updated.
            r (int): The rank of the adapter. Should be a positive integer or 0.
            lora_alpha (float): The alpha value for Lora scaling.
            lora_dropout (float): The dropout probability for Lora. Should be greater than 0.0.
            init_lora_weights (bool): If True, initializes Lora weights.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            ValueError: If the value of 'r' is less than 0, a ValueError is raised.
        """
        if r < 0:
            # note: r == 0 is allowed for AdaLora, see #1539
            raise ValueError(f"`r` should be a positive integer or 0, but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        # Right singular vectors
        if r > 0:
            weight_A = ops.randn((r, self.in_features))
            weight_E = ops.randn((r, 1))
            weight_B = ops.randn((self.out_features, r))
        else:
            rank_idx = Tensor([False])
            weight_A = ops.randn((1, self.in_features))
            weight_E = ops.randn((1, 1))
            weight_B = ops.randn((self.out_features, 1))
            weight_A = weight_A[rank_idx, :]
            weight_E = weight_E[rank_idx, :]
            weight_B = weight_B[:, rank_idx]
        self.lora_A.update({adapter_name: Parameter(weight_A)})
        # Singular values
        self.lora_E.update({adapter_name: Parameter(weight_E)})
        # Left singular vectors
        self.lora_B.update({adapter_name: Parameter(weight_B)})
        # The current rank
        self.ranknum.update({adapter_name: Parameter(Tensor(float(r)), requires_grad=False)})
        self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else float(r)
        if init_lora_weights and r > 0:
            self.reset_lora_parameters(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            nn.init.zeros_(self.lora_E[adapter_name])
            nn.init.normal_(self.lora_A[adapter_name], mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B[adapter_name], mean=0.0, std=0.02)


class SVDLinear(nn.Module, AdaLoraLayer):
    "SVD-based adaptation by a dense layer"
    # SVD-based adaptation by a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        r"""
        Initializes an instance of the SVDLinear class.
        
        Args:
            self: The object itself.
            base_layer (nn.Module): The base layer of the SVDLinear model.
            adapter_name (str): The name of the adapter.
            r (int, optional): The number of singular values to keep. Defaults to 0.
            lora_alpha (int, optional): The alpha value for the LORA algorithm. Defaults to 1.
            lora_dropout (float, optional): The dropout rate for the LORA algorithm. Defaults to 0.0.
            fan_in_fan_out (bool, optional): Indicates whether to use fan-in/fan-out scaling. Defaults to False.
            init_lora_weights (bool, optional): Indicates whether to initialize the LORA weights. Defaults to True.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None. This method does not raise any exceptions.
        """
        super().__init__()
        AdaLoraLayer.__init__(self, base_layer)
        # Freezing the pre-trained weight matrix
        self.get_base_layer().requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            base_layer = self.get_base_layer()
            if active_adapter in self.lora_A.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not ops.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> Tensor:
        r"""
        Calculates the delta weight for a given adapter in the SVDLinear class.
        
        Args:
            self (SVDLinear): An instance of the SVDLinear class.
            adapter: The adapter index for which the delta weight needs to be calculated. 
        
        Returns:
            Tensor: A tensor representing the delta weight for the specified adapter.
        
        Raises:
            None.
        
        This method calculates the delta weight for a specific adapter in the SVDLinear class. The delta weight is computed using the following formula:
        
            delta_weight = transpose(self.lora_B[adapter] @ (self.lora_A[adapter] * self.lora_E[adapter]), self.fan_in_fan_out) * self.scaling[adapter] / (self.ranknum[adapter] + 1e-05)
        
        The method returns the calculated delta weight as a Tensor object.
        """
        return (
            transpose(self.lora_B[adapter] @ (self.lora_A[adapter] * self.lora_E[adapter]), self.fan_in_fan_out)
            * self.scaling[adapter]
            / (self.ranknum[adapter] + 1e-5)
        )

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        r"""Constructs a tensor using the SVDLinear method.
        
        Args:
            self: An instance of the SVDLinear class.
            x (Tensor): The input tensor for the forward method.
        
        Returns:
            Tensor: The forwarded tensor.
        
        Raises:
            None.
        """
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                lora_E = self.lora_E[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                ranknum = self.ranknum[active_adapter] + 1e-5

                x = x.to(lora_A.dtype)
                result += (dropout(x) @ (lora_A * lora_E).T @ lora_B.T) * scaling / ranknum

        return result

    def __repr__(self) -> str:
        r"""
        This method returns a string representation of the object.
        
        Args:
            self: SVDLinear instance. Represents the current instance of the SVDLinear class.
        
        Returns:
            str: A string representation of the object, prefixed with 'adalora.'.
        
        Raises:
            No specific exceptions are raised within this method.
        """
        rep = super().__repr__()
        return "adalora." + rep


class RankAllocator:
    """
    The RankAllocator for AdaLoraModel. Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        model: the model that we apply AdaLoRA to.

    """
    def __init__(self, model, peft_config, adapter_name):
        """
        Initializes a RankAllocator instance.
        
        Args:
            self: The RankAllocator instance.
            model: The model to be used for rank allocation.
            peft_config: The PEFT configuration object containing beta1 and beta2 values.
            adapter_name: The name of the adapter.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            AssertionError: If the beta1 or beta2 values in peft_config are not within the range (0, 1).
        """
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        assert self.beta1 > 0 and self.beta1 < 1
        assert self.beta2 > 0 and self.beta2 < 1

        self.reset_ipt()
        self._set_budget_scheduler(model)

    def set_total_step(self, total_step):
        r"""
        Sets the total number of steps in the RankAllocator.
        
        Args:
            self (RankAllocator): The RankAllocator object.
            total_step (int): The total number of steps in the RankAllocator. It specifies the maximum number of steps that can be allocated.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            None.
        
        """
        self.peft_config.total_step = total_step

    def reset_ipt(self):
        r"""
        Resets the 'ipt' attribute, along with its associated attributes 'exp_avg_ipt' and 'exp_avg_unc', in the RankAllocator class.
        
        Args:
            self: An instance of the RankAllocator class.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            This method does not raise any exceptions.
        """
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

    def _set_budget_scheduler(self, model):
        r"""
        This method '_set_budget_scheduler' belongs to the class 'RankAllocator' and is responsible for setting up the budget scheduler based on the provided model.
        
        Args:
            self: Instance of the RankAllocator class. It is used to access and modify the attributes and methods of the class.
            model: An object representing the model. The method iterates through the parameters and names of the model to calculate the initial budget 'init_bgt' and create a set of names 'name_set'.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            No specific exceptions are documented to be raised by this method. However, potential exceptions could arise from working with the input parameters or during the iteration process.
        """
        self.init_bgt = 0
        self.name_set = set()
        for n, p in model.parameters_and_names():
            if f"lora_A.{self.adapter_name}" in n:
                self.init_bgt += p.data.shape[0]
                self.name_set.add(n.replace("lora_A", "%s"))
        self.name_set = sorted(self.name_set)
        # The total final rank budget
        self.target_bgt = self.peft_config.target_r * len(self.name_set)

    def budget_schedule(self, step: int):
        r"""
        This method calculates the budget and mask indicator based on the given step value.
        
        Args:
            self (RankAllocator): The instance of the RankAllocator class.
            step (int): The current step for which the budget and mask indicator need to be calculated. It should be a non-negative integer.
        
        Returns:
            tuple: A tuple containing the calculated budget and mask indicator. The budget is an integer representing the budget value, and the mask indicator is a boolean indicating whether the mask should be
applied.
        
        Raises:
            No specific exceptions are raised by this method.
        """
        tinit = self.peft_config.tinit
        tfinal = self.peft_config.tfinal
        total_step = self.peft_config.total_step
        # Initial warmup
        if step <= tinit:
            budget = self.init_bgt
            mask_ind = False
        # Final fine-tuning
        elif step > total_step - tfinal:
            budget = self.target_bgt
            mask_ind = True
        else:
            # Budget decreasing with a cubic scheduler
            mul_coeff = 1 - (step - tinit) / (total_step - tfinal - tinit)
            budget = int((self.init_bgt - self.target_bgt) * (mul_coeff**3) + self.target_bgt)
            mask_ind = step % self.peft_config.deltaT == 0
        return budget, mask_ind

    def update_ipt(self, model,gradient):
        r"""
        This method updates the importance parameter table (ipt) for the given model using the provided gradient.
        
        Args:
            self: The instance of the RankAllocator class.
            model: The model for which the importance parameter table is being updated.
                   Type: model object
                   Purpose: To access the parameters and names of the model for updating the ipt.
                   Restrictions: None
            gradient: The gradient to be used for updating the ipt.
                      Type: gradient object
                      Purpose: To calculate the importance parameter table based on the gradient.
                      Restrictions: None
        
        Returns:
            None. The method does not return any value.
        
        Raises:
            None
        """
        # Update the sensitivity and uncertainty for every weight
        for n, p in model.parameters_and_names():
            if "lora_" in n and self.adapter_name in n:
                if n not in self.ipt:
                    grad = get_grad(gradient, p)
                    self.ipt[n] = ops.zeros_like(p)
                    self.exp_avg_ipt[n] = ops.zeros_like(p)
                    self.exp_avg_unc[n] = ops.zeros_like(p)
                    self.ipt[n] = (p * grad).abs()
                    # Sensitivity smoothing
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                    # Uncertainty quantification
                    self.exp_avg_unc[n] = (
                        self.beta2 * self.exp_avg_unc[n] + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                    )

    def _element_score(self, n):
        r"""
        This method calculates the element score based on the exponential average input and exponential average uncertainty values.
        
        Args:
            self (RankAllocator): The instance of the RankAllocator class.
            n (int): The index of the element for which the score needs to be calculated. It should be a non-negative integer.
        
        Returns:
            None: This method does not return any value explicitly but calculates the element score based on the input parameters.
        
        Raises:
            - IndexError: If the index 'n' is out of bounds or negative.
            - TypeError: If the input values are not of the expected types.
        """
        return self.exp_avg_ipt[n] * self.exp_avg_unc[n]

    def _combine_ipt(self, ipt_E, ipt_AB):
        r"""
        This method combines two input arrays, ipt_E and ipt_AB, into a single array and returns the resulting sum.
        
        Args:
            self (object): The instance of the RankAllocator class.
            ipt_E (array-like): An array containing elements to be combined. It should be a 1-dimensional array.
            ipt_AB (array-like): An array containing elements to be combined. It should be a 2-dimensional array.
        
        Returns:
            array-like: A 1-dimensional array containing the sum of the elements from ipt_E and ipt_AB.
        
        Raises:
            ValueError: If ipt_AB is not a 2-dimensional array.
            TypeError: If ipt_E is not a valid array-like object.
            TypeError: If ipt_AB is not a valid array-like object.
            ValueError: If the shapes of ipt_E and ipt_AB are not compatible for addition.
        
        """
        ipt_AB = ipt_AB.sum(axis=1, keepdims=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_budget(self, model, budget):
        r""" 
        The 'mask_to_budget' method in the class 'RankAllocator' calculates a mask threshold based on the given budget and applies the threshold to mask certain parameters in the model.
        
        Args:
            self: The instance of the RankAllocator class.
            model: A model object representing the neural network model to be processed. It is expected to have a method 'parameters_and_names()' and 'masked_fill()' to access and modify model parameters.
            budget: An integer representing the budget for masking parameters. It restricts the number of parameters that can be masked based on their importance scores.
        
        Returns:
            None: This method does not return any value. It modifies the model parameters in place based on the calculated mask threshold and budget.
        
        Raises:
            - KeyError: If the adapter name specified in the method is not found in the model parameters.
            - ValueError: If the budget provided is not a positive integer.
            - TypeError: If the input model is not the expected type or format. 
            - RuntimeError: If there are any runtime errors during the execution of the method.
        """
        value_ipt = {}
        vector_ipt = {}
        triplet_ipt = {}
        # Get the importance score for A, E, B
        for n, p in model.parameters_and_names():
            if f"lora_A.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = ops.mean(entry_ipt, dim=1, keepdim=True)
                name_m = n.replace("lora_A", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"lora_B.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = ops.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
                name_m = n.replace("lora_B", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"lora_E.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                name_m = n.replace("lora_E", "%s")
                value_ipt[name_m] = entry_ipt

        all_score = []
        # Calculate the score for each triplet
        for name_m in vector_ipt:
            ipt_E = value_ipt[name_m]
            ipt_AB = ops.cat(vector_ipt[name_m], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_m % "lora_E"
            triplet_ipt[name_E] = sum_ipt.view(-1, 1)
            all_score.append(sum_ipt.view(-1))

        # Get the threshold by ranking ipt
        mask_threshold = ops.topk(
            ops.cat(all_score),
            k=self.init_bgt - budget,
            largest=False
        )[0][self.init_bgt - budget-1].item()

        rank_pattern = {}
        # Mask the unimportant triplets
        for n, p in model.parameters_and_names():
            if f"lora_E.{self.adapter_name}" in n:
                p.masked_fill(triplet_ipt[n] <= mask_threshold, 0.0)
                rank_pattern[n] = (~(triplet_ipt[n] <= mask_threshold)).view(-1).asnumpy().tolist()
        return rank_pattern

    def update_and_allocate(self, model, global_step, gradient, force_mask=False):
        r"""
        This method updates the model and allocates budget based on the global step and gradient information.
        
        Args:
        - self: Reference to the current instance of the class.
        - model: The model to be updated and allocated the budget.
        - global_step: The current global step of the training process.
        - gradient: The gradient information used for updating the model.
        - force_mask: A boolean flag indicating whether to force the masking operation. Default is False.
        
        Returns:
        - budget: The allocated budget for the current step.
        - rank_pattern: The rank pattern based on the budget allocation, or None if no masking is needed.
        
        Raises:
        - No specific exceptions are raised by this method.
        """
        # # Update the importance score and allocate the budget
        if global_step < self.peft_config.total_step - self.peft_config.tfinal:
            self.update_ipt(model,gradient)
        budget, mask_ind = self.budget_schedule(global_step)
        # Allocate the budget according to importance scores
        if mask_ind or force_mask:
            rank_pattern = self.mask_to_budget(model, budget)
        else:
            rank_pattern = None
        return budget, rank_pattern

    def mask_using_rank_pattern(self, model, rank_pattern):
        r""" 
            Applies a mask to the model parameters based on the provided rank pattern.
        
            Args:
                self (RankAllocator): The instance of the RankAllocator class.
                model: The model containing the parameters to be masked.
                rank_pattern: A dictionary containing the rank pattern used for masking the parameters.
                              The keys of the dictionary represent parameter names, and the corresponding
                              values are the mask patterns.
        
            Returns:
                None. The method modifies the model parameters in-place.
        
            Raises:
                None.
        """
        def mask_using_rank_pattern(self, model, rank_pattern):
            """
            Applies a mask to the model parameters based on the provided rank pattern.
        
            Args:
                self (RankAllocator): The instance of the RankAllocator class.
                model: The model containing the parameters to be masked.
                rank_pattern: A dictionary containing the rank pattern used for masking the parameters.
                              The keys of the dictionary represent parameter names, and the corresponding
                              values are the mask patterns.
        
            Returns:
                None. The method modifies the model parameters in-place.
        
            Raises:
                None.
            """
        # Mask the unimportant triplets
        is_adapter_name_truncated = False
        if self.adapter_name not in next(iter(rank_pattern.keys())):
            is_adapter_name_truncated = True

        for n, p in model.parameters_and_names():
            if f"lora_E.{self.adapter_name}" in n:
                key = n if not is_adapter_name_truncated else n.replace(f".{self.adapter_name}", "")
                mask = Tensor(rank_pattern[key]).unsqueeze(-1)
                p.masked_fill(~mask.bool(), 0.0)
