from mindspore.ops.auto_generate import gen_ops_prim
from mindspore.ops.auto_generate import pyboost_inner_prim
from mindspore._c_expression import _empty_instance

gen_ops_list = list(filter(lambda s: s.startswith("pyboost"), dir(gen_ops_prim)))
pyboost_inner_list = list(filter(lambda s: s.endswith("_impl"), dir(pyboost_inner_prim)))

__all__ = []

for pyboost_op_name in gen_ops_list:
    op_name = pyboost_op_name.replace('pyboost_', '') + '_op'
    func_name = op_name.replace('_op', '')
    op_instance = getattr(gen_ops_prim, op_name, None)
    if op_instance is not None:
        __all__.append(func_name)
        globals()[func_name] = getattr(gen_ops_prim, op_name).__class__().set_device('Ascend')

for op_name in pyboost_inner_list:
    func_name = op_name.replace('_impl', '')
    __all__.append(func_name)
    globals()[func_name] = getattr(pyboost_inner_prim, op_name).__class__()

def empty(*args, **kwargs):
    return _empty_instance(*args, **kwargs, device='Ascend')

__all__.append('empty')
