# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""
Time series distributional output classes and utilities.
"""
from typing import Callable, Dict, Optional, Tuple
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore.nn.probability.distribution import (
    Distribution,
    Normal,
    StudentT,
    TransformedDistribution,
    # AffineTransform,
    # Independent,
    # NegativeBinomial,
)
from mindspore.nn.probability.bijector import ScalarAffine as AffineTransform
import numpy as np

class AffineTransformed(TransformedDistribution):
    '''
    # todo 
    '''
    def __init__(self, base_distribution: Distribution, loc=None, scale=None, event_dim=0):
        """
        Initializes an instance of the AffineTransformed class.
        
        Args:
            self: The instance of the class.
            base_distribution (Distribution): The base distribution to be transformed.
            loc (float, optional): The location parameter for the affine transformation. Defaults to None.
            scale (float, optional): The scale parameter for the affine transformation. Defaults to None.
            event_dim (int, optional): The number of dimensions in the event space. Defaults to 0.
        
        Returns:
            None
        
        Raises:
            None
        """
        self.scale = 1.0 if scale is None else scale
        self.loc = 0.0 if loc is None else loc
        super().__init__(AffineTransform(shift=self.loc, scale=self.scale), base_distribution)

    def _set_attr_for_tensor(self, name, value):
        """
        Sets an attribute for a tensor in the AffineTransformed class.
        
        Args:
            self (object): The instance of the AffineTransformed class.
            name (str): The name of the attribute to be set.
            value (Any): The value to be assigned to the attribute.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        object.__setattr__(self, name, value)

    def mean(self):
        """
        Returns the mean of the distribution.
        """
        return self.base_dist.mean * self.scale + self.loc

    def variance(self):
        """
        Returns the variance of the distribution.
        """
        return self.base_dist.variance * self.scale**2

    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()


class ParameterProjection(nn.Cell):
    """
    # todo
    """
    def __init__(
        self, in_features: int, args_dim: Dict[str, int], domain_map: Callable[..., Tuple[mindspore.Tensor]], **kwargs
    ) -> None:
        """
        Initializes an instance of the ParameterProjection class.
        
        Args:
            self: The object instance.
            in_features (int): The number of input features.
            args_dim (Dict[str, int]): A dictionary containing the dimensions for each argument.
            domain_map (Callable[..., Tuple[mindspore.Tensor]]): A callable that maps the input domain to a tuple of tensors.
            
        Returns:
            None
            
        Raises:
            None
        """
        super().__init__(**kwargs)
        self.args_dim = args_dim
        self.proj = nn.CellList([nn.Dense(in_features, dim) for dim in args_dim.values()])
        self.domain_map = domain_map

    def construct(self, x: mindspore.Tensor) -> Tuple[mindspore.Tensor]:
        """
        Constructs the parameter projection using the provided input tensor.
        
        Args:
            self (ParameterProjection): An instance of the ParameterProjection class.
            x (mindspore.Tensor): The input tensor representing the data to be projected.
            
        Returns:
            Tuple[mindspore.Tensor]:
                A tuple containing the projected tensor(s) after applying parameter projection.
        
        Raises:
            TypeError: If the input tensor 'x' is not of type mindspore.Tensor.
            ValueError: If there is an issue with the domain mapping operation.
        """
        params_unbounded = [proj(x) for proj in self.proj]

        return self.domain_map(*params_unbounded)


class LambdaLayer(nn.Cell):
    """
    #todo
    """
    def __init__(self, function):
        """
        Initializes an instance of the LambdaLayer class.
        
        Args:
            self (LambdaLayer): The instance of the LambdaLayer class.
            function (function): The function that will be stored in the LambdaLayer instance.
        
        Returns:
            None.
        
        Raises:
            None
        """
        super().__init__()
        self.function = function

    def construct(self, x, *args):
        """
        Constructs a LambdaLayer object.
        
        Args:
            self (LambdaLayer): The current instance of the LambdaLayer class.
            x: The input parameter for the lambda function.
                - Type: Any
                - Purpose: Represents the input value for the lambda function.
            *args: Variable length argument list.
                - Type: Any
                - Purpose: Additional arguments that can be passed to the lambda function.

        Returns:
            None.

        Raises:
            None.
        """
        return self.function(x, *args)


class DistributionOutput:
    """
    # todo
    """
    distribution_class: type
    in_features: int
    args_dim: Dict[str, int]

    def __init__(self, dim: int = 1) -> None:
        """
        Initializes an instance of the DistributionOutput class.

        Args:
            self (DistributionOutput): The instance of the class.
            dim (int, optional): The dimension of the output. Defaults to 1.

        Returns:
            None.

        Raises:
            None.
        """
        self.dim = dim
        self.args_dim = {k: dim * self.args_dim[k] for k in self.args_dim}

    def _base_distribution(self, distr_args):
        """
        Method: _base_distribution

        This method is a helper method for the DistributionOutput class.
        It creates an instance of the distribution class specified by the class variable 'distribution_class' using
        the provided 'distr_args' and returns it.

        Args:
            self:
                A reference to the current instance of the DistributionOutput class.

                - Type: DistributionOutput
                - Purpose: Allows access to the class's variables and methods.
                - Restrictions: None.
            distr_args:
                A list of arguments to be passed to the distribution class constructor.

                - Type: list
                - Purpose: Specifies the arguments required to instantiate the distribution class.
                - Restrictions: The number and types of arguments must be compatible with the distribution class constructor.

        Returns:
            None:
                This method does not return any value.

                - Type: None
                - Purpose: The method is used for its side effects, specifically, creating an instance of the
                distribution class.
        
        Raises:
            None.
        """
        #if self.dim == 1:
        return self.distribution_class(*distr_args)
        #return Independent(self.distribution_class(*distr_args), 1)

    def distribution(
        self,
        distr_args,
        loc: Optional[mindspore.Tensor] = None,
        scale: Optional[mindspore.Tensor] = None,
    ) -> Distribution:
        r"""
        # todo
        """
        distr = self._base_distribution(distr_args)
        if loc is None and scale is None:
            return distr
        return AffineTransformed(distr, loc=loc, scale=scale, event_dim=self.event_dim)

    @property
    def event_shape(self) -> Tuple:
        r"""
        Shape of each individual event contemplated by the distributions that this object constructs.
        """
        return () if self.dim == 1 else (self.dim,)

    @property
    def event_dim(self) -> int:
        r"""
        Number of event dimensions, i.e., length of the `event_shape` tuple, of the distributions that this object
        constructs.
        """
        return len(self.event_shape)

    @property
    def value_in_support(self) -> float:
        r"""
        A float that will have a valid numeric value when computing the log-loss of the corresponding distribution. By
        default 0.0. This value will be used when padding data series.
        """
        return 0.0

    def get_parameter_projection(self, in_features: int) -> nn.Cell:
        r"""
        Return the parameter projection layer that maps the input to the appropriate parameters of the distribution.
        """
        return ParameterProjection(
            in_features=in_features,
            args_dim=self.args_dim,
            domain_map=LambdaLayer(self.domain_map),
        )

    def domain_map(self, *args: mindspore.Tensor):
        r"""
        Converts arguments to the right shape and domain. The domain depends on the type of distribution, while the
        correct shape is obtained by reshaping the trailing axis in such a way that the returned tensors define a
        distribution of the right event_shape.
        """
        raise NotImplementedError()

    @staticmethod
    def squareplus(x: mindspore.Tensor) -> mindspore.Tensor:
        r"""
        Helper to map inputs to the positive orthant by applying the square-plus operation. Reference:
        https://twitter.com/jon_barron/status/1387167648669048833
        """
        return (x + ops.sqrt(ops.square(x) + 4.0)) / 2.0


class StudentTOutput(DistributionOutput):
    """
    Student-T distribution output class.
    """
    args_dim: Dict[str, int] = {"df": 1, "loc": 1, "scale": 1}
    distribution_class: type = StudentT

    @classmethod
    def domain_map(cls, df: mindspore.Tensor, loc: mindspore.Tensor, scale: mindspore.Tensor):
        """
        Method to perform domain mapping on input tensors.
        
        Args:
            cls (class): The class reference.
            df (mindspore.Tensor): Input tensor representing degrees of freedom.
                It should be a 1D tensor.
            loc (mindspore.Tensor): Input tensor representing location parameter.
                It should be a 1D tensor.
            scale (mindspore.Tensor): Input tensor representing scale parameter.
                It should be a 1D tensor.
        
        Returns:
            None.
        
        Raises:
            ValueError: If the input tensors are not in the expected format.
            TypeError: If the input tensors have incompatible data types.
            AssertionError: If the input tensors fail the domain mapping conditions.
        """
        scale = cls.squareplus(scale).clamp(
            mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(scale.dtype)).eps))
        df = 2.0 + cls.squareplus(df)
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)


class NormalOutput(DistributionOutput):
    """
    Normal distribution output class.
    """
    args_dim: Dict[str, int] = {"loc": 1, "scale": 1}
    distribution_class: type = Normal

    @classmethod
    def domain_map(cls, loc: mindspore.Tensor, scale: mindspore.Tensor):
        """
        This method 'domain_map' in the class 'NormalOutput' processes the location and scale tensors for domain mapping.
        
        Args:
            cls (class): The class reference.
            loc (mindspore.Tensor): The input location tensor.
                Purpose: Represents the location data.
                Restrictions: Should be of type mindspore.Tensor.
            scale (mindspore.Tensor): The input scale tensor.
                Purpose: Represents the scale data.
                Restrictions: Should be of type mindspore.Tensor.
        
        Returns:
            None:
                - Purpose: The method does not return any specific value.
                It processes the input tensors and modifies them in place.
        
        Raises:
            None
        """
        scale = cls.squareplus(scale).clamp_min(np.finfo(mindspore.dtype_to_nptype(scale.dtype)).eps)
        return loc.squeeze(-1), scale.squeeze(-1)
