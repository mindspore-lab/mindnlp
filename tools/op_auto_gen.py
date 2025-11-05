import re
import inspect
import importlib
import argparse

import mindspore
from mindspore import ops
from mindspore.ops.auto_generate import gen_ops_prim
from mindspore.ops.auto_generate import pyboost_inner_prim

def camel_to_snake_case_improved(camel_case_str):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_case_str)
    snake_case_str = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return snake_case_str

op_func_no_init = '''
{name}_op = {op}()
def {name}(*args):
    {name}_op = hook_call({name}_op)
    return {name}_op(*args)
'''

op_func_with_init = '''
def {name}(*args):
    op = _get_cache_prim({op})(*args[-{idx}:])
    op = hook_call(op)
    return op(*args[:-{idx}])
'''

def gen_legacy_op(gen_file):
    op_list = list(filter(lambda s: s[0].isupper(), ops.operations.__all__))
    grad_op = list(filter(lambda s: s[0].isupper(), dir(mindspore.ops.operations._grad_ops)))

    op_dict = {
        'mindspore.ops.operations._grad_ops': grad_op,
        'mindspore.ops.operations': op_list
    }

    with open(gen_file, 'w') as f:
        f.write("from mindspore.ops.operations import *\n"
                "from mindspore.ops.operations._grad_ops import *\n"
                "from mindspore.ops._primitive_cache import _get_cache_prim\n\n")
        for op_module, op_list in op_dict.items():
            for old_op_name in op_list:
                if old_op_name in ['P', 'Print', 'Assert', 'Custom', 'CustomOpBuilder', 'DataType', 'ReduceOp', 'TBERegOp', 'Tensor']:
                    continue

                op_mod = importlib.import_module(op_module)
                ops_class = getattr(op_mod, old_op_name, None)
                init_signature = inspect.signature(ops_class.__init__)
                name = camel_to_snake_case_improved(old_op_name)
                if len(init_signature.parameters) > 1:
                    init_args = list(init_signature.parameters.keys())
                    init_args.pop(0)
                    code = op_func_with_init.format(name=name, op=old_op_name, idx=len(init_args))

                else:
                    code = op_func_no_init.format(name=name, op=old_op_name)
                f.write(code + '\n')
    f.close()

def gen_aclnn_op(gen_file):
    gen_ops_list = list(filter(lambda s: s.startswith("pyboost"), dir(gen_ops_prim)))
    pyboost_inner_list = list(filter(lambda s: s.endswith("_impl"), dir(pyboost_inner_prim)))

    with open(gen_file, 'w') as f:
        f.write("from mindspore.ops.auto_generate.gen_ops_prim import *\n"
                "from mindspore.ops.auto_generate.pyboost_inner_prim import *\n\n")

        for pyboost_op_name in gen_ops_list:
            op_name = pyboost_op_name.replace('pyboost_', '') + '_op'
            op_instance = getattr(gen_ops_prim, op_name, None)
            if op_instance is not None:
                f.write(f"{op_name} = {getattr(gen_ops_prim, op_name).__class__.__name__}().set_device('Ascend')\n\n")

        # for op_name in pyboost_inner_list:
        #     f.write(f"{op_name} = {getattr(pyboost_inner_prim, op_name).__class__.__name__}()\n\n")
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 添加位置参数
    parser.add_argument('output_file', type=str)
    parser.add_argument('--op_type', type=str, default='legacy', required=False ,choices=['legacy', 'pyboost'])


    args = parser.parse_args()
    print(args)
    if args.op_type == 'legacy':
        gen_legacy_op(args.output_file)
    elif args.op_type == 'pyboost':
        gen_aclnn_op(args.output_file)
        