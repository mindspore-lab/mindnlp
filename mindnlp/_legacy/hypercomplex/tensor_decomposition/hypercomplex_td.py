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

class LinearTDLayer(nn.Cell):
    def __init__(self, in_ch, out_ch, bias, rank):
        super().__init__()
        self.linear0 = Dense(in_ch, rank, has_bias=False)
        self.linear1 = Dense(rank, out_ch, has_bias=bias)
    
    def construct(self, u: mindspore.Tensor, v: mindspore.Tensor):
        u, v = self.linear0(u, v)
        return self.linear1(u, v)

def decompose_linear_parameters(param, threshold = 0.5, count_interations = 10):
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
    total_params = 0
    for name, parameter in model.parameters_and_names():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        total_params+=param

    print(f"Total Params: {total_params}")
    return total_params
