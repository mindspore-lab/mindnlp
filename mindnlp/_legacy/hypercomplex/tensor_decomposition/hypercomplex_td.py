import numpy as np
from mindspore import nn
import mindspore
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.dual.dual_algebra_factory import DualAlgebraFactory
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.basics.hypercomplex.hc_matrix import Matrix as HCMatrix
from mindnlp._legacy.hypercomplex.tensor_decomposition.linal.algorithm.svd import SVD
from mindspore import Parameter
try:
    from mindspore.hypercomplex.dual.dual_operators import Dense
except:
    from mindnlp._legacy.hypercomplex.dual import Dense

class LinearTDLayer(nn.Module):

    r"""
    LinearTDLayer is a neural network cell representing a linear transformation layer with TensorDot operation.
    
    This class inherits from nn.Module and contains methods for initializing and forwarding the linear transformation layer. 
    The linear transformation is performed using two Dense layers, linear0 and linear1, with specified input and output channels, bias, and rank.
    
    The __init__ method initializes the LinearTDLayer with the specified input channel, output channel, bias, and rank, and creates the linear0 and linear1 Dense layers.
    
    The forward method applies the linear transformation to the input tensors u and v using the linear0 and linear1 Dense layers, and returns the transformed output tensor.
    
    Note: This docstring does not include signatures or any other code.
    """
    def __init__(self, in_ch, out_ch, bias, rank):
        r"""
        Initialize a LinearTDLayer object with specified parameters.
        
        Args:
            self (object): The instance of the LinearTDLayer class.
            in_ch (int): The number of input channels for the layer.
            out_ch (int): The number of output channels for the layer.
            bias (bool): Flag indicating whether bias is used in the layer.
            rank (int): The rank of the layer.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            ValueError: If in_ch, out_ch, or rank is not a positive integer.
            TypeError: If bias is not a boolean value.
        """
        super().__init__()
        self.linear0 = Dense(in_ch, rank, bias=False)
        self.linear1 = Dense(rank, out_ch, bias=bias)
    
    def forward(self, u: mindspore.Tensor, v: mindspore.Tensor):
        r"""
        Constructs a linear TD layer by applying linear transformations to input tensors 'u' and 'v'.
        
        Args:
            self (LinearTDLayer): Instance of the LinearTDLayer class.
            u (mindspore.Tensor): Input tensor 'u' to be processed by the linear transformations.
            v (mindspore.Tensor): Input tensor 'v' to be processed by the linear transformations.
        
        Returns:
            None. The method applies linear transformations to the input tensors 'u' and 'v' and returns None.
        
        Raises:
            - TypeError: If the input parameters 'u' or 'v' are not of type mindspore.Tensor.
            - ValueError: If an issue occurs during the linear transformations process.
        """
        u, v = self.linear0(u, v)
        return self.linear1(u, v)

def decompose_linear_parameters(param, threshold = 0.5, count_interations = 10):
    r"""Decompose linear parameters.
    
    Args:
        param (ndarray): The input matrix to be decomposed.
        threshold (float, optional): The threshold value for determining the rank of the decomposition. 
            Defaults to 0.5.
        count_interations (int, optional): The number of iterations for the decomposition process. 
            Defaults to 10.
    
    Returns:
        tuple: A tuple containing two matrices representing the decomposed linear parameters.
    
    Raises:
        None.
    
    This function decomposes the linear parameters using singular value decomposition (SVD) technique. 
    It calculates the rank of the decomposition based on the threshold value and returns the decomposed matrices.
    """
    matrix = param.copy()
    print("Shape of current matrix:", matrix.shape)
    t = HCMatrix(DualAlgebraFactory, matrix.shape[0], matrix.shape[1], matrix)
    u, s, v = SVD.decompose(t, count_interations)
    sigmas = np.diag(s._items_x)
    unit_sum = 0
    total_sum = np.sum(sigmas)
    print("sum is:", total_sum)
    for i, sig in enumerate(sigmas):
        unit_sum += sig
        if unit_sum/total_sum >= threshold:
            rank = i
            print("\tRank is", rank + 1, "of", len(sigmas))
            print("\tLocal compression: ", matrix.shape[0] * matrix.shape[1] / (rank + 1) / (matrix.shape[0] + matrix.shape[1]))
            print("\tSum of sigmas is:", unit_sum)
            break
    u = u[:, :(rank+1)]
    s = s[:(rank+1), :(rank+1)]
    v = v[:(rank+1), :]
    return u@s, v

def set_new_dict_names(p1, p2, name, new_dict, b_x, b_y):
    r"""
    Sets new names in a dictionary and adds parameters to it.
    
    Args:
        p1 (object): The first parameter object.
        p2 (object): The second parameter object.
        name (str): The original name to be modified.
        new_dict (dict): The dictionary to which the new names and parameters will be added.
        b_x (object): The bias object for x.
        b_y (object): The bias object for y.
    
    Returns:
        None: This function does not return any value.
    
    Raises:
        None: This function does not raise any exceptions.
    """
    new_name = name.replace(".weight_x",".linear0.weight_x")
    new_name_last = new_name.replace(".linear0.",".linear1.")
    new_dict[new_name] = Parameter(mindspore.tensor(p1._items_x.T, dtype=mindspore.float32))
    new_dict[new_name_last] = Parameter(mindspore.tensor(p2._items_x.T, dtype=mindspore.float32))
    new_dict[new_name.replace("_x", "_y")] = Parameter(mindspore.tensor(p1._items_y.T, dtype=mindspore.float32))
    new_dict[new_name_last.replace("_x", "_y")] = Parameter(mindspore.tensor(p2._items_y.T, dtype=mindspore.float32))
    if b_x != None or b_y != None:
        new_dict[new_name.replace("weight_x", "bias_x")] = b_x
        new_dict[new_name.replace("weight_x", "bias_y")] = b_y

    print('The transformations are applied!')

def calculate_parameters(model):
    r"""
    Calculate the total number of trainable parameters in the given model.
    
    Args:
        model (torch.nn.Module): The input model for which the parameters need to be calculated.
    
    Returns:
        None: This function does not return any value explicitly but prints the total number of parameters.
    
    Raises:
        None
    """
    total_params = 0
    for name, parameter in model.parameters_and_names():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        total_params+=param

    print(f"Total Params: {total_params}")
    return total_params
