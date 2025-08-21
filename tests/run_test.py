import os
import sys

import pytest
import mindspore
import mindnlp
from mindnlp import transformers

mindspore.set_context(pynative_synchronize=True)

def run_tests():
    """
    使用pytest.main()执行测试，支持所有pytest命令行参数
    用法: python run_test.py [pytest参数] [测试路径]
    示例: 
        python run_test.py -v tests/
        python run_test.py -k "login" tests/test_auth.py
        python run_test.py tests/test_api.py::TestLogin::test_invalid_credentials
    """
    # 获取命令行参数（排除脚本名本身）
    pytest_args = sys.argv[1:]
    # not support sdpa/loss.backward/torchscript/torch.fx/torch.compile
    skip_ut = "not sdpa " \
        "and not headmasking " \
        "and not gradient_checkpointing " \
        "and not retain_grad " \
        "and not data_parallel " \
        "and not model_parallelism " \
        "and not with_static_cache " \
        "and not compile " \
        "and not compilation " \
        "and not torchscript " \
        "and not torch_fx " \
        "and not test_wrong_device_map " \
        "and not test_layerwise_casting " \
        "and not test_flex_attention " \
        "and not offload " \
        "and not global_device"

    pytest_args.extend(["--ignore-glob=test_modeling_flax_*.py"])
    pytest_args.extend(['-k', skip_ut])
    if not pytest_args:
        print("未提供参数，默认运行当前目录下所有测试")
        print("使用示例: python run_test.py -v tests/")

    # 执行测试并获取退出码
    exit_code = pytest.main(pytest_args)
    
    # 根据退出码处理结果
    if exit_code == 0:
        print("\n✅ 所有测试通过!")
    else:
        print(f"\n❌ 测试失败，退出码: {exit_code}")
        print("常见退出码: 0=通过, 1=失败, 2=中断, 3=内部错误, 4=命令行错误")
    
    return exit_code

if __name__ == "__main__":
    # 执行并返回系统退出码
    sys.exit(run_tests())