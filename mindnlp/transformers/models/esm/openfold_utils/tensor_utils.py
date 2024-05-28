# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""tensor utils"""
from functools import partial
from typing import Any, Callable, Dict, List, Type, TypeVar, Union, overload

import mindspore
from mindspore import ops


def add(m1: mindspore.Tensor, m2: mindspore.Tensor, inplace: bool) -> mindspore.Tensor:
    """
    Function to add two tensors either in place or creating a new one.
    
    Args:
        m1 (mindspore.Tensor): The first tensor to be added.
        m2 (mindspore.Tensor): The second tensor to be added.
        inplace (bool): If True, the addition is done in place on m1. If False, a new tensor is created.
    
    Returns:
        mindspore.Tensor: The resulting tensor after addition.
    
    Raises:
        TypeError: If m1 or m2 is not of type mindspore.Tensor.
    """
    # The first operation in a checkpoint can't be in-place, but it's
    # nice to have in-place addition during inference. Thus...
    if not inplace:
        m1 = m1 + m2
    else:
        m1 += m2

    return m1


def permute_final_dims(tensor: mindspore.Tensor, inds: List[int]) -> mindspore.Tensor:
    """
    Permute the final dimensions of a given tensor.
    
    Args:
        tensor (mindspore.Tensor): The input tensor to permute the final dimensions.
        inds (List[int]): The list of indices specifying the new order of the final dimensions. 
    
    Returns:
        mindspore.Tensor: The permuted tensor with the final dimensions rearranged according to the provided indices.
    
    Raises:
        None.
    """
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: mindspore.Tensor, no_dims: int) -> mindspore.Tensor:
    """
    Flattens the final dimensions of a given tensor.
    
    Args:
        t (mindspore.Tensor): The input tensor.
            - Type: mindspore.Tensor
            - Purpose: Specifies the tensor to be flattened.
        no_dims (int): The number of dimensions to be flattened.
            - Type: int
            - Purpose: Specifies the number of dimensions to be flattened from the end of the tensor.
            - Restrictions: It must be a non-negative integer.
    
    Returns:
        mindspore.Tensor: The flattened tensor.
            - Type: mindspore.Tensor
            - Purpose: Represents the flattened tensor.
    
    Raises:
        None.
    
    """
    return t.reshape(t.shape[:-no_dims] + (-1,))


def masked_mean(mask: mindspore.Tensor, value: mindspore.Tensor, dim: int, eps: float = 1e-4) -> mindspore.Tensor:
    """
    Calculates the masked mean along a specified dimension of a given tensor.
    
    Args:
        mask (mindspore.Tensor): A tensor representing the mask. It should have the same shape as the `value` tensor.
        value (mindspore.Tensor): A tensor containing the values to be masked.
        dim (int): An integer specifying the dimension along which the mean is calculated.
        eps (float, optional): A small epsilon value added to the denominator to avoid division by zero. Default is 0.0001.
    
    Returns:
        mindspore.Tensor: A tensor representing the masked mean along the specified dimension.
    
    Raises:
        TypeError: If `mask` or `value` is not of type mindspore.Tensor.
        ValueError: If `mask` and `value` do not have the same shape.
        ValueError: If `dim` is out of range for the shape of the input tensors.
        ValueError: If `eps` is not a positive float value.
    
    Note:
        The masked mean is calculated by multiplying the `mask` tensor with the `value` tensor, summing the masked values along the specified dimension, and dividing it by the sum of the mask values along the
same dimension plus a small epsilon value.
    
    Example:
        mask = mindspore.Tensor([[1, 1, 1], [0, 1, 0]])
        value = mindspore.Tensor([[3, 4, 5], [6, 7, 8]])
        dim = 1
        result = masked_mean(mask, value, dim)
        # Returns: mindspore.Tensor([4.0, 7.0])
    """
    mask = mask.expand(*value.shape)
    return ops.sum(mask * value, dim=dim) / (eps + ops.sum(mask, dim=dim))


def pts_to_distogram(
    pts: mindspore.Tensor, min_bin = 2.3125, max_bin = 21.6875, no_bins: int = 64
) -> mindspore.Tensor:
    """
    Converts a set of points to a distogram representation.
    
    Args:
        pts (mindspore.Tensor): The input tensor containing the points. The shape of the tensor is expected to be (N, D),
                                where N represents the number of points and D represents the number of dimensions.
        min_bin (float, optional): The minimum value of the bins. Defaults to 2.3125.
        max_bin (float, optional): The maximum value of the bins. Defaults to 21.6875.
        no_bins (int, optional): The number of bins. Defaults to 64.
    
    Returns:
        mindspore.Tensor: A tensor representing the distogram. The shape of the tensor is (N, N), where N is the number
                          of points. Each element represents the bin index to which the distance between two points belongs.
    
    Raises:
        None.
    """
    boundaries = ops.linspace(min_bin, max_bin, no_bins - 1)
    dists = ops.sqrt(ops.sum((pts.unsqueeze(-2) - pts.unsqueeze(-3)) ** 2, dim=-1))
    return ops.bucketize(dists, boundaries)


def dict_multimap(fn: Callable[[list], Any], dicts: List[dict]) -> dict:
    """
    Apply a function to corresponding values in multiple dictionaries.
    
    Args:
        fn (Callable[[list], Any]): The function to apply to the values.
        dicts (List[dict]): A list of dictionaries containing the values to be processed.
    
    Returns:
        dict: A new dictionary where each key is processed by the function fn on corresponding values from input dictionaries.
    
    Raises:
        IndexError: If the input list of dictionaries is empty.
        KeyError: If a key is missing in any of the input dictionaries.
        TypeError: If the function fn receives an argument of incorrect type.
    """
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if isinstance(v, dict):
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def one_hot(x: mindspore.Tensor, v_bins: mindspore.Tensor) -> mindspore.Tensor:
    """
    Converts an input tensor into a one-hot encoded tensor based on the provided bins.
    
    Args:
        x (mindspore.Tensor): The input tensor to be encoded.
        v_bins (mindspore.Tensor): The tensor representing the bins for one-hot encoding.
    
    Returns:
        mindspore.Tensor: The one-hot encoded tensor with the same shape as the input tensor.
    
    Raises:
        None.
    
    """
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = ops.argmin(ops.abs(diffs), axis=-1)
    return ops.one_hot(am, len(v_bins)).float()


def batched_gather(data: mindspore.Tensor, inds: mindspore.Tensor, dim: int = 0, no_batch_dims: int = 0) -> mindspore.Tensor:
    """
    This function performs batched gathering of elements from the input data tensor based on the provided indices.
    
    Args:
        data (mindspore.Tensor): The input data tensor from which elements are gathered.
        inds (mindspore.Tensor): The indices along the specified dimension for gathering elements.
        dim (int, optional): The dimension along which the gathering operation is performed. Defaults to 0.
        no_batch_dims (int, optional): The number of batch dimensions in the input data tensor. Defaults to 0.
    
    Returns:
        mindspore.Tensor: A new tensor containing the gathered elements based on the provided indices.
    
    Raises:
        TypeError: If the input data, indices, or dimension arguments are not of the correct types.
        ValueError: If the dimensions of the input tensors are not compatible for the batched gather operation.
    """
    ranges: List[Union[slice, mindspore.Tensor]] = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = ops.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims: List[Union[slice, mindspore.Tensor]] = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    # Matt note: Editing this to get around the behaviour of using a list as an array index changing
    # in recent Numpy versions
    return data[tuple(ranges)]


T = TypeVar("T")


# With tree_map, a poor man's JAX tree_map
def dict_map(
    fn: Callable[[T], Any], dic: Dict[Any, Union[dict, list, tuple, T]], leaf_type: Type[T]
) -> Dict[Any, Union[dict, list, tuple, Any]]:
    """
    Recursively applies a function to all leaf values in a dictionary.
    
    Args:
        fn (Callable[[T], Any]): The function to apply to each leaf value in the dictionary.
            It should take a single argument of type T and return a value of any type.
        dic (Dict[Any, Union[dict, list, tuple, T]]): The input dictionary to be processed.
            It may contain nested dictionaries, lists, tuples, and leaf values of type T.
        leaf_type (Type[T]): The type of leaf values in the dictionary that the function should be applied to.
            Only leaf values of this type will be processed. Any other values will be left unchanged.
    
    Returns:
        Dict[Any, Union[dict, list, tuple, Any]]: A new dictionary with the same structure as the input dictionary,
            but with the function applied to all leaf values of type T.
    
    Raises:
        None
    
    Note:
        This function recursively traverses the input dictionary and applies the given function to any leaf value of type T.
        The function is not applied to nested dictionaries, lists, or tuples.
    """
    new_dict: Dict[Any, Union[dict, list, tuple, Any]] = {}
    for k, v in dic.items():
        if isinstance(v, dict):
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


@overload
def tree_map(fn: Callable[[T], Any], tree: T, leaf_type: Type[T]) -> Any:
    """
    Apply a function to each element of a tree structure.
    
    Args:
        fn: A callable that takes a single parameter of type 'T' and returns a value of any type. This function will be applied to each element of the tree.
        tree: The tree structure to apply the function to. The tree can be of any type, but it should be compatible with the 'fn' parameter.
        leaf_type: The type of the leaf elements in the tree. This is used to restrict the 'tree' parameter to be of the same type. 
    
    Returns:
        The result of applying the 'fn' function to each element of the 'tree'. The return type can be of any type.
    
    Raises:
        None.
    
    """


@overload
def tree_map(fn: Callable[[T], Any], tree: dict, leaf_type: Type[T]) -> dict:
    """
    Applies a function to each leaf value in a dictionary tree and returns a new dictionary with the modified values.
    
    Args:
        fn (Callable[[T], Any]): A function that takes a leaf value of type T and returns a modified value.
        tree (dict): The dictionary tree to be traversed and modified.
        leaf_type (Type[T]): The type of the leaf values in the dictionary tree.
    
    Returns:
        dict: A new dictionary tree with the same structure as the input tree, but with the leaf values modified by the provided function.
    
    Raises:
        None.
    
    Note:
        This function operates on the leaf values of the dictionary tree, which are defined as the values that are not themselves dictionaries.
    """


@overload
def tree_map(fn: Callable[[T], Any], tree: list, leaf_type: Type[T]) -> list:
    """
    Apply a function to each element of a tree structure.
    
    Args:
        fn (Callable[[T], Any]): A function to apply to each element of the tree structure.
        tree (list): The tree structure to map the function onto.
        leaf_type (Type[T]): The type of leaf elements in the tree structure.
    
    Returns:
        list: A new tree structure with the function applied to each element.
    
    Raises:
        None.
    """


@overload
def tree_map(fn: Callable[[T], Any], tree: tuple, leaf_type: Type[T]) -> tuple:
    """
    Args:
        fn (Callable[[T], Any]): A function to be applied to each leaf node in the tree.
        tree (tuple): The input tree structure where the function will be applied to each leaf node.
        leaf_type (Type[T]): The type of the leaf nodes in the tree.
    
    Returns:
        tuple: A new tree structure with the function applied to each leaf node.
    
    Raises:
        None
    """


def tree_map(fn, tree, leaf_type):
    """
    Apply a given function to each element in a tree-like structure.
    
    Args:
        fn (function): The function to apply to each element in the tree.
        tree (dict, list, tuple): The tree-like structure to apply the function to.
        leaf_type (type): The type of leaf elements in the tree.
    
    Returns:
        None: This function does not return a value.
    
    Raises:
        ValueError: If the tree contains an element that is not supported by the function.
    """
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple(tree_map(fn, x, leaf_type) for x in tree)
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        print(type(tree))
        raise ValueError("Not supported")


tensor_tree_map = partial(tree_map, leaf_type=mindspore.Tensor)
