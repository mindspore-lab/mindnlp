# Copyright 2022 Huawei Technologies Co., Ltd
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
"""attention module"""
import math
from typing import Optional

import mindspore
from mindnlp.core.nn import Parameter
from mindnlp.core import nn, ops
from .utils import masked_softmax, tiny_value_of_dtype, get_combined_dim,  combine_tensors_and_multiply

class Attention(nn.Module):
    """
    An `Attention` takes two inputs: a (batched) vector and a matrix, plus an optional mask on the
    rows of the matrix.  We compute the similarity between the vector and each row in the matrix,
    and then (optionally) perform a softmax over rows using those computed similarities.


    Inputs:

    - vector: shape `(batch_size, embedding_dim)`
    - matrix: shape `(batch_size, num_rows, embedding_dim)`
    - matrix_mask: shape `(batch_size, num_rows)`, specifying which rows are just padding.

    Output:

    - attention: shape `(batch_size, num_rows)`.

    # Parameters

    normalize : `bool`, optional (default = `True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self._normalize = normalize

    def forward(
        self, vector: mindspore.Tensor, matrix: mindspore.Tensor, matrix_mask: mindspore.Tensor = None
    ) -> mindspore.Tensor:
        similarities = self._forward_internal(vector, matrix)
        if self._normalize:
            return masked_softmax(similarities, matrix_mask)
        else:
            return similarities

    def _forward_internal(self, vector: mindspore.Tensor, matrix: mindspore.Tensor) -> mindspore.Tensor:
        raise NotImplementedError


class AdditiveAttention(Attention):
    """
    Computes attention between a vector and a matrix using an additive attention function.  This
    function has two matrices `W`, `U` and a vector `V`. The similarity between the vector
    `x` and the matrix `y` is computed as `V tanh(Wx + Uy)`.

    This attention is often referred as concat or additive attention. It was introduced in
    [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al, 2015)]
    (https://api.semanticscholar.org/CorpusID:11212020).

    Registered as an `Attention` with name "additive".

    # Parameters

    vector_dim : `int`, required
        The dimension of the vector, `x`, described above.  This is `x.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_dim : `int`, required
        The dimension of the matrix, `y`, described above.  This is `y.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    normalize : `bool`, optional (default = `True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(self, vector_dim: int, matrix_dim: int, normalize: bool = True) -> None:
        super().__init__(normalize)
        self._w_matrix = Parameter(mindspore.Tensor(vector_dim, vector_dim))
        self._u_matrix = Parameter(mindspore.Tensor(matrix_dim, vector_dim))
        self._v_vector = Parameter(mindspore.Tensor(vector_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._w_matrix)
        nn.init.xavier_uniform_(self._u_matrix)
        nn.init.xavier_uniform_(self._v_vector)

    def _forward_internal(self, vector: mindspore.Tensor, matrix: mindspore.Tensor) -> mindspore.Tensor:
        intermediate = vector.matmul(self._w_matrix).unsqueeze(1) + matrix.matmul(self._u_matrix)
        intermediate = ops.tanh(intermediate)
        return intermediate.matmul(self._v_vector).squeeze(2)


class BilinearAttention(Attention):
    """
    Computes attention between a vector and a matrix using a bilinear attention function.  This
    function has a matrix of weights `W` and a bias `b`, and the similarity between the vector
    `x` and the matrix `y` is computed as `x^T W y + b`.

    Registered as an `Attention` with name "bilinear".

    # Parameters

    vector_dim : `int`, required
        The dimension of the vector, `x`, described above.  This is `x.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_dim : `int`, required
        The dimension of the matrix, `y`, described above.  This is `y.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    activation : `Activation`, optional (default=`linear`)
        An activation function applied after the `x^T W y + b` calculation.  Default is
        linear, i.e. no activation.
    normalize : `bool`, optional (default=`True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(
        self,
        vector_dim: int,
        matrix_dim: int,
        activation: None,
        normalize: bool = True,
    ) -> None:
        super().__init__(normalize)
        self._weight_matrix = Parameter(mindspore.Tensor(vector_dim, matrix_dim))
        self._bias = Parameter(mindspore.Tensor(1))
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._weight_matrix)
        nn.init.zeros_(self._bias)

    def _forward_internal(self, vector: mindspore.Tensor, matrix: mindspore.Tensor) -> mindspore.Tensor:
        intermediate = vector.mm(self._weight_matrix).unsqueeze(1)
        return self._activation(intermediate.bmm(matrix.swapaxes(1, 2)).squeeze(1) + self._bias)


class CosineAttention(Attention):
    """
    Computes attention between a vector and a matrix using cosine similarity.

    Registered as an `Attention` with name "cosine".
    """

    def _forward_internal(self, vector: mindspore.Tensor, matrix: mindspore.Tensor) -> mindspore.Tensor:
        a_norm = vector / (
            vector.norm(p=2, dim=-1, keepdim=True) + tiny_value_of_dtype(vector.dtype)
        )
        b_norm = matrix / (
            matrix.norm(p=2, dim=-1, keepdim=True) + tiny_value_of_dtype(matrix.dtype)
        )
        return ops.bmm(a_norm.unsqueeze(dim=1), b_norm.swapaxes(-1, -2)).squeeze(1)


class DotProductAttention(Attention):
    """
    Computes attention between a vector and a matrix using dot product.

    Reference: [Attention Is All You Need (Vaswani et al, 2017)]
    (https://api.semanticscholar.org/CorpusID:13756489)

    Registered as an `Attention` with name "dot_product".
    """

    def _forward_internal(self, vector: mindspore.Tensor, matrix: mindspore.Tensor) -> mindspore.Tensor:
        return matrix.bmm(vector.unsqueeze(-1)).squeeze(-1)

class LinearAttention(Attention):
    """
    This `Attention` module performs a dot product between a vector of weights and some
    combination of the two input vectors, followed by an (optional) activation function.  The
    combination used is configurable.

    If the two vectors are `x` and `y`, we allow the following kinds of combinations : `x`,
    `y`, `x*y`, `x+y`, `x-y`, `x/y`, where each of those binary operations is performed
    elementwise.  You can list as many combinations as you want, comma separated.  For example, you
    might give `x,y,x*y` as the `combination` parameter to this class.  The computed similarity
    function would then be `w^T [x; y; x*y] + b`, where `w` is a vector of weights, `b` is a
    bias parameter, and `[;]` is vector concatenation.

    Note that if you want a bilinear similarity function with a diagonal weight matrix W, where the
    similarity function is computed as `x * w * y + b` (with `w` the diagonal of `W`), you can
    accomplish that with this class by using "x*y" for `combination`.

    Registered as an `Attention` with name "linear".

    # Parameters

    tensor_1_dim : `int`, required
        The dimension of the first tensor, `x`, described above.  This is `x.size()[-1]` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    tensor_2_dim : `int`, required
        The dimension of the second tensor, `y`, described above.  This is `y.size()[-1]` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    combination : `str`, optional (default=`"x,y"`)
        Described above.
    activation : `Activation`, optional (default=`linear`)
        An activation function applied after the `w^T * [x;y] + b` calculation.  Default is
        linear, i.e. no activation.
    normalize : `bool`, optional (default=`True`)
    """

    def __init__(
        self,
        tensor_1_dim: int,
        tensor_2_dim: int,
        combination: str = "x,y",
        activation: nn.Module = None,
        normalize: bool = True,
    ) -> None:
        super().__init__(normalize)
        self._combination = combination
        combined_dim = get_combined_dim(combination, [tensor_1_dim, tensor_2_dim])
        self._weight_vector = Parameter(mindspore.Tensor(combined_dim))
        self._bias = Parameter(mindspore.Tensor(1))
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self._weight_vector.shape[0] + 1))
        nn.init.uniform_(self._weight_vector, -std, std)
        self._bias.data.fill_(0)

    def _forward_internal(self, vector: mindspore.Tensor, matrix: mindspore.Tensor) -> mindspore.Tensor:
        combined_tensors = combine_tensors_and_multiply(
            self._combination, [vector.unsqueeze(1), matrix], self._weight_vector
        )
        return self._activation(combined_tensors.squeeze(1) + self._bias)

class ScaledDotProductAttention(DotProductAttention):
    """
    Computes attention between two tensors using scaled dot product.
    # Reference: [Attention Is All You Need (Vaswani et al, 2017)]
    # (https://api.semanticscholar.org/CorpusID:13756489)

    Registered as an `Attention` with name "scaled_dot_product".

    # Parameters

    scaling_factor : `int`, required
        The similarity score is scaled down by the `scaling_factor`.
    normalize : `bool`, optional (default=`True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(self, scaling_factor: Optional[int] = None, normalize: bool = True) -> None:
        super().__init__(normalize)
        self.scaling_factor = scaling_factor

    def _forward_internal(self, vector: mindspore.Tensor, matrix: mindspore.Tensor) -> mindspore.Tensor:
        scores = super()._forward_internal(vector, matrix)
        scaling_factor = self.scaling_factor or matrix.size(-1)
        scores = scores / math.sqrt(scaling_factor)
        return scores

__all__ = [
    "ScaledDotProductAttention",
    "DotProductAttention",
    "LinearAttention",
    "BilinearAttention",
    "AdditiveAttention",
    "CosineAttention",
]
