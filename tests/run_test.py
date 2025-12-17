import os
import sys

# Add src directory to Python path to allow importing packages
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pytest
import mindspore
from mindspore.common.api import _pynative_executor

import mindnlp
from mindnlp import transformers

if mindspore.get_context("device_target") == "GPU":
    os.environ["TRANSFORMERS_TEST_DEVICE"] = "cuda"
elif mindspore.get_context("device_target") == "Ascend":
    os.environ["TRANSFORMERS_TEST_DEVICE"] = "npu"
elif mindspore.get_context("device_target") == "CPU":
    os.environ["TRANSFORMERS_TEST_DEVICE"] = "cpu"
else:
    raise ValueError(f"Unsupported device target: {mindspore.get_context('device_target')}")


if os.environ.get('TEST_LAUNCH_BLOCKING', 'True').strip().lower() == 'true':
    mindspore.runtime.launch_blocking()


class PytestHooks:
    """
    Custom pytest plugin with hooks
    You can add any pytest hooks here, such as:
    - pytest_configure
    - pytest_sessionstart
    - pytest_sessionfinish
    - pytest_runtest_setup
    - pytest_runtest_teardown
    - pytest_collection_modifyitems
    - etc.
    """
    
    def pytest_configure(self, config):
        """Called after command line options have been parsed."""
        print("🔧 pytest_configure hook called")
        # You can register additional plugins here if needed
        # config.pluginmanager.register(SomeOtherPlugin(), "plugin_name")
    
    def pytest_sessionstart(self, session):
        """Called after the Session object has been created."""
        print("🚀 pytest_sessionstart hook called")
        # Note: session.items is not available yet at this point
        # Use pytest_collection_finish or pytest_collection_modifyitems to access items
    
    def pytest_sessionfinish(self, session, exitstatus):
        """Called after whole test run finished, right before returning the exit status."""
        print("🏁 pytest_sessionfinish hook called")
        print(f"   Exit status: {exitstatus}")
    
    def pytest_runtest_setup(self, item):
        """Called before running a test item."""
        # This is called before each test
        _pynative_executor.set_grad_flag(True)
    
    def pytest_runtest_teardown(self, item):
        """Called after running a test item."""
        # This is called after each test
        pass
    
    def pytest_collection_modifyitems(self, config, items):
        """Called after collection has been performed."""
        # You can modify the test items here
        # For example, add markers, skip tests, etc.
        pass


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
        "and not data_parallel " \
        "and not model_parallelism " \
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

    # 创建自定义hook插件
    hooks_plugin = PytestHooks()
    
    # 执行测试并获取退出码
    # 通过 plugins 参数传递自定义hook插件
    exit_code = pytest.main(pytest_args, plugins=[hooks_plugin])
    
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