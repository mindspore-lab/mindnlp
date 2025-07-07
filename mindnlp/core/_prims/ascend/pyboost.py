from mindspore.ops import Primitive
from mindspore.ops.auto_generate import gen_ops_prim
from mindspore.ops.auto_generate.gen_ops_prim import *
from mindspore._c_expression import pyboost_cast, pyboost_zeros, pyboost_ones, pyboost_empty, \
    pyboost_reduce_max, pyboost_reduce_min, pyboost_reduce_all, pyboost_reduce_all
from mindspore.ops.operations.manually_defined.ops_def import Cast, Zeros, Ones
from mindspore.common.api import _pynative_executor

pyboost_list = list(filter(lambda s: s.startswith("pyboost"), dir(gen_ops_prim)))


pyboost_func = '''
def {name}(*args):
    return {pyboost}({op}, args)
'''

__all__ = []

for op_name in pyboost_list:
    op = getattr(gen_ops_prim, op_name)
    func_name = op_name.replace('pyboost_', '') + '_npu'
    prim_op = func_name.replace('_npu', '_op')
    if not hasattr(gen_ops_prim, prim_op):
        continue
    __all__.append(func_name)
    globals()[prim_op] = getattr(gen_ops_prim, prim_op).__class__().set_device('Ascend')
    exec(pyboost_func.format(name=func_name, pyboost=op_name, op=prim_op), globals())

cast_op = Cast().set_device('Ascend')
def cast_npu(*args):
    return pyboost_cast(cast_op, args)

__all__.append('cast_npu')

def empty_npu(size, dtype):
    return pyboost_empty([size, dtype, 'Ascend'])

__all__.append('empty_npu')

zeros_op = Zeros().set_device('Ascend')
def zeros_npu(*args):
    return pyboost_zeros(zeros_op, args)

__all__.append('zeros_npu')

ones_op = Ones().set_device('Ascend')
def ones_npu(*args):
    return pyboost_ones(ones_op, args)

__all__.append('ones_npu')


squeeze_op = Squeeze().set_device('Ascend')
def squeeze_npu(*args):
    return pyboost_squeeze(squeeze_op, args)

__all__.append('squeeze_npu')

stack_ext_op = StackExt().set_device('Ascend')
def stack_ext_npu(*args):
    return pyboost_stack_ext(stack_ext_op, args)

__all__.append('stack_ext_npu')

tile_op = Primitive('Tile').set_device('Ascend')
def tile_npu(*args):
    return pyboost_tile(tile_op, args)

__all__.append('tile_npu')

greater_equal_op = GreaterEqual().set_device('Ascend')
def greater_equal_npu(*args):
    return pyboost_greater_equal(greater_equal_op, args)

__all__.append('greater_equal_npu')

isclose_op = IsClose().set_device('Ascend')
def isclose_npu(*args):
    return pyboost_isclose(isclose_op, args)

__all__.append('isclose_npu')

reduce_max_op = ReduceMax().set_device('Ascend')
def reduce_max_npu(*args):
    return pyboost_reduce_max(reduce_max_op, args)

__all__.append('reduce_max_npu')

reduce_min_op = ReduceMin().set_device('Ascend')
def reduce_min_npu(*args):
    return pyboost_reduce_min(reduce_min_op, args)

__all__.append('reduce_min_npu')

reduce_all_op = ReduceAll().set_device('Ascend')
def reduce_all_npu(*args):
    return pyboost_reduce_all(reduce_all_op, args)

__all__.append('reduce_all_npu')

reduce_any_op = ReduceAny().set_device('Ascend')
def reduce_any_npu(*args):
    return pyboost_reduce_any(reduce_any_op, args)

__all__.append('reduce_any_npu')

unique_consecutive_op = UniqueConsecutive().set_device('Ascend')
def unique_consecutive_npu(*args):
    return pyboost_unique_consecutive(unique_consecutive_op, args)

__all__.append('unique_consecutive_npu')

nan_to_num_op = NanToNum().set_device('Ascend')
def nan_to_num_npu(*args):
    return pyboost_nan_to_num(nan_to_num_op, args)

__all__.append('nan_to_num_npu')


softmax_op = Softmax().set_device('Ascend')
def softmax_npu(*args):
    return pyboost_softmax(softmax_op, args)

__all__.append('softmax_npu')

broadcast_to_op = Primitive('BroadcastTo').set_device('Ascend')
def broadcast_to_npu(*args):
    return pyboost_broadcast_to(broadcast_to_op, args)

__all__.append('broadcast_to_npu')

triu_op = Triu().set_device('Ascend')
def triu_npu(*args):
    return pyboost_triu(triu_op, args)

__all__.append('triu_npu')

tril_ext_op = TrilExt().set_device('Ascend')
def tril_ext_npu(*args):
    return pyboost_tril_ext(triu_op, args)

__all__.append('tril_ext_npu')

search_sorted_op = SearchSorted().set_device('Ascend')
def search_sorted_npu(*args):
    return pyboost_searchsorted(search_sorted_op, args)

__all__.append('search_sorted_npu')

roll_op = Primitive('Roll').set_device('Ascend')
def roll_npu(*args):
    return pyboost_roll(roll_op, args)

__all__.append('roll_npu')

meshgrid_op = Meshgrid().set_device('Ascend')
def meshgrid_npu(*args):
    return pyboost_meshgrid(meshgrid_op, args)

__all__.append('meshgrid_npu')

reverse_v2_op = Primitive('ReverseV2').set_device('Ascend')
def reverse_v2_npu(*args):
    return pyboost_reverse_v2(reverse_v2_op, args)

__all__.append('reverse_v2_npu')

hard_shrink_op = HShrink().set_device('Ascend')
def hard_shrink_npu(*args):
    return pyboost_hshrink(hard_shrink_op, args)

__all__.append('hard_shrink_npu')

concat_op = Concat().set_device('Ascend')
def concat_npu(*args):
    return pyboost_concat(concat_op, args)

__all__.append('concat_npu')

rms_norm_op = RmsNorm().set_device('Ascend')
def rms_norm_npu(*args):
    return pyboost_rms_norm(rms_norm_op, args)

__all__.append('rms_norm_npu')

flash_attention_score_op = Primitive('FlashAttentionScore').set_device('Ascend')
def flash_attention_score_npu(*args):
    return pyboost_flash_attention_score(flash_attention_score_op, args)

__all__.append('flash_attention_score_npu')

argmax_with_value_op = ArgMaxWithValue().set_device('Ascend')
def argmax_with_value_npu(*args):
    return pyboost_argmax_with_value(argmax_with_value_op, args)

__all__.append('argmax_with_value_npu')

argmin_with_value_op = ArgMinWithValue().set_device('Ascend')
def argmin_with_value_npu(*args):
    return pyboost_argmin_with_value(argmin_with_value_op, args)

__all__.append('argmin_with_value_npu')
