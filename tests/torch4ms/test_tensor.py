# Copyright 2025 Google LLC
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

import unittest
import numpy as np
import torch
from mindspore import Tensor as ms_Tensor
from mindspore import Parameter

from torch4ms.tensor import Tensor, Environment, OperatorNotFound
from torch4ms import config


class TestTensor(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        self.test_config = config.Configuration()
        self.test_config.debug_print_each_op = False
        self.test_config.use_ms_graph_mode = False
        self.env = Environment(self.test_config)
    
    def test_tensor_initialization(self):
        """测试Tensor类的初始化功能"""
        # 测试从NumPy数组创建
        np_array = np.array([1, 2, 3], dtype=np.float32)
        tensor = Tensor(np_array, self.env)
        self.assertIsInstance(tensor._elem, ms_Tensor)
        self.assertEqual(tensor.shape, (3,))
        
        # 测试从MindSpore Tensor创建
        ms_tensor = ms_Tensor(np_array)
        tensor = Tensor(ms_tensor, self.env)
        self.assertIs(tensor._elem, ms_tensor)
        
        # 测试requires_grad参数
        tensor = Tensor(np_array, self.env, requires_grad=True)
        self.assertIsInstance(tensor._elem, Parameter)
        
    def test_tensor_properties(self):
        """测试Tensor类的属性"""
        np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tensor = Tensor(np_array, self.env)
        
        # shape ndim dim dtype
        self.assertEqual(tensor.shape, (2, 2))
        self.assertEqual(tensor.ndim, 2)
        self.assertEqual(tensor.dim(), 2)
        dtype_str = str(tensor.dtype).lower()
        self.assertTrue('float32' in dtype_str or 'float' in dtype_str)

        # 通过numpy()方法后的数组类型来验证
        numpy_dtype = tensor.numpy().dtype
        self.assertTrue(np.issubdtype(numpy_dtype, np.floating))
        
        # 测试device属性
        device_str = str(tensor.device)
        self.assertTrue('CPU' in device_str or 'cpu' in device_str)
    
    def test_tensor_conversion(self):
        """测试Tensor类的转换方法"""
        np_array = np.array([1, 2, 3], dtype=np.float32)
        tensor = Tensor(np_array, self.env)
        
        # 测试numpy()方法
        numpy_result = tensor.numpy()
        self.assertIsInstance(numpy_result, np.ndarray)
        np.testing.assert_allclose(numpy_result, np_array)
        
        # 测试mindspore()方法
        ms_result = tensor.mindspore()
        self.assertIsInstance(ms_result, ms_Tensor)
        
        # 测试tolist()方法
        list_result = tensor.tolist()
        self.assertEqual(list_result, [1.0, 2.0, 3.0])
    
    def test_tensor_methods(self):
        """测试Tensor类的基本方法"""
        np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tensor = Tensor(np_array, self.env)
        
        # 测试flatten方法
        flattened = tensor.flatten()
        self.assertIsInstance(flattened, Tensor)
        self.assertEqual(flattened.shape, (4,))
        
        # 测试detach方法
        detached = tensor.detach()
        self.assertIsInstance(detached, Tensor)
        
        # 测试__setitem__方法
        tensor_copy = Tensor(np_array.copy(), self.env)
        tensor_copy[0, 0] = 10
        self.assertEqual(tensor_copy._elem[0, 0].asnumpy(), 10)
    
    def test_apply_mindspore(self):
        """测试apply_mindspore方法"""
        np_array = np.array([1, 2, 3], dtype=np.float32)
        tensor = Tensor(np_array, self.env)
        
        # 测试应用MindSpore操作
        from mindspore import ops
        result = tensor.apply_mindspore(ops.square)
        self.assertIsInstance(result, Tensor)
        expected = np.array([1, 4, 9], dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected)
    
    def test_environment_basic(self):
        """测试Environment类的基本功能"""
        # 上下文管理器
        with self.env as env:
            self.assertTrue(env.enabled)
        self.assertFalse(self.env.enabled)
        self.env.manual_seed(1)
        self.assertEqual(self.env.param.prng, 1)
    
    def test_tensor_conversion_functions(self):
        """测试张量转换函数"""
        # 测试ms2t_iso: MindSpore Tensor->torch4ms Tensor
        # 在MindSpore中，数据类型直接通过mindspore模块访问
        import mindspore
        ms_tensor = ms_Tensor([1, 2, 3], dtype=mindspore.float32)
        converted = self.env.ms2t_iso(ms_tensor)
        self.assertIsInstance(converted, Tensor)
        
        # 测试嵌套结构的转换
        nested = [ms_tensor, {'key': ms_tensor}]
        converted_nested = self.env.ms2t_iso(nested)
        self.assertIsInstance(converted_nested[0], Tensor)
        self.assertIsInstance(converted_nested[1]['key'], Tensor)
        
        # 测试t2ms_iso: torch4ms Tensor->MindSpore Tensor
        torch4ms_tensor = Tensor([1, 2, 3], self.env)
        ms_result = self.env.t2ms_iso(torch4ms_tensor)
        self.assertIsInstance(ms_result, ms_Tensor)
    
    def test_device_check(self):
        """测试设备检查功能"""
        # 测试_should_use_torch4ms_tensor方法
        self.assertFalse(self.env._should_use_torch4ms_tensor('gpu'))
        self.assertFalse(self.env._should_use_torch4ms_tensor('npu'))
        
        # 修改配置后测试
        self.env.config.treat_cuda_as_mindspore_device = True
        self.assertTrue(self.env._should_use_torch4ms_tensor('cuda'))
        
        # 重置为False，确认行为正确
        self.env.config.treat_cuda_as_mindspore_device = False
        self.assertFalse(self.env._should_use_torch4ms_tensor('cuda'))
    
    def test_override_property(self):
        """测试属性覆盖功能"""
        original_prng = self.env.param.prng
        
        # 使用上下文管理器临时覆盖属性
        with self.env.override_property(prng=12345):
            self.assertEqual(self.env.param.prng, 12345)
        
        # 上下文结束后应恢复原值
        self.assertEqual(self.env.param.prng, original_prng)
    
    def test_operator_not_found_exception(self):
        """测试OperatorNotFound异常"""
        # 尝试分发一个不存在的操作
        with self.assertRaises(OperatorNotFound):
            self.env.dispatch("non_existent_operator")
    
    def test_override_op_definition(self):
        """测试覆盖操作定义"""
        # 定义一个简单的自定义操作实现
        def custom_implementation(x):
            return x * 2
        self.env.override_op_definition("custom_op", custom_implementation)
        
        # 验证操作被正确注册
        try:
            op = self.env._get_op_or_decomp("custom_op")
            self.assertEqual(op.func, custom_implementation)
            self.assertTrue(op.is_user_defined)
        except OperatorNotFound:
            self.fail("自定义操作未被正确注册")
    
    def test_tensor_conversion_with_pytorch(self):
        """测试与PyTorch张量的转换"""
        # 创建PyTorch张量
        torch_tensor = torch.tensor([1.0, 2.0, 3.0])
        
        # 创建MindSpore张量
        from torch4ms.ops import mappings
        ms_tensor = mappings.t2ms(torch_tensor)
        
        # 转换为torch4ms张量
        torch4ms_tensor = Tensor(ms_tensor, self.env)
        
        # 测试转换回PyTorch张量
        converted_torch = mappings.ms2t(torch4ms_tensor.mindspore())

        np.testing.assert_allclose(torch_tensor.numpy(), converted_torch.numpy())
    
    def test_complex_tensor_operations(self):
        """测试复杂张量操作"""
        a = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32), self.env)
        b = Tensor(np.array([[5, 6], [7, 8]], dtype=np.float32), self.env)
        
        # 测试矩阵乘法
        from mindspore import ops
        c = a.apply_mindspore(ops.matmul, b._elem)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        print(c)
        np.testing.assert_allclose(c.numpy(), expected)
        
        # 测试元素级乘法
        d = a.apply_mindspore(ops.mul, b._elem)
        expected = np.array([[5, 12], [21, 32]], dtype=np.float32)
        np.testing.assert_allclose(d.numpy(), expected)
    
    def test_tensor_arithmetics(self):
        """测试张量算术运算"""
        tensor = Tensor(np.array([1, 2, 3], dtype=np.float32), self.env)
        
        # 加法
        result = tensor.apply_mindspore(lambda x: x + 1)
        expected = np.array([2, 3, 4], dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected)
        
        # 减法
        result = tensor.apply_mindspore(lambda x: x - 1)
        expected = np.array([0, 1, 2], dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected)
        
        # 乘法
        result = tensor.apply_mindspore(lambda x: x * 2)
        expected = np.array([2, 4, 6], dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected)
        
        # 除法
        result = tensor.apply_mindspore(lambda x: x / 2)
        expected = np.array([0.5, 1.0, 1.5], dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected)
    
    def test_tensor_reductions(self):
        """测试张量归约操作"""
        tensor = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), self.env)
        
        # 求和
        from mindspore import ops
        sum_result = tensor.apply_mindspore(ops.reduce_sum)
        self.assertEqual(sum_result.numpy(), 21.0)
        
        # 按维度求和
        sum_axis0 = tensor.apply_mindspore(ops.reduce_sum, 0)
        expected = np.array([5, 7, 9], dtype=np.float32)
        np.testing.assert_allclose(sum_axis0.numpy(), expected)
        
        # 平均值
        mean_result = tensor.apply_mindspore(ops.reduce_mean)
        self.assertEqual(mean_result.numpy(), 3.5)
    
    def test_to_copy_functionality(self):
        """测试_to_copy方法功能"""
        tensor = Tensor(np.array([1, 2, 3], dtype=np.float32), self.env)
        
        # 测试数据类型转换
        from mindspore import float64
        new_tensor = self.env._to_copy(tensor, float64, None)
        self.assertEqual(new_tensor.dtype, float64)
        
        # 注意：设备转换测试可能需要实际的GPU环境，这里仅测试CPU到CPU的情况
        cpu_tensor = self.env._to_copy(tensor, None, 'cpu')
        self.assertIsInstance(cpu_tensor, Tensor)
    
    def test_tree_conversion_functions(self):
        """测试树结构转换函数"""
        import mindspore
        ms_tensor = ms_Tensor([1, 2, 3], dtype=mindspore.float32)
        complex_structure = {
            'a': ms_tensor,
            'b': [ms_tensor, {'c': ms_tensor}],
            'd': (ms_tensor, ms_tensor)
        }
        
        # 测试ms2t_iso转换整个树结构
        converted = self.env.ms2t_iso(complex_structure)
        self.assertIsInstance(converted['a'], Tensor)
        self.assertIsInstance(converted['b'][0], Tensor)
        self.assertIsInstance(converted['b'][1]['c'], Tensor)
        self.assertIsInstance(converted['d'][0], Tensor)
        
        # 测试t2ms_iso转换回MindSpore张量
        ms_converted = self.env.t2ms_iso(converted)
        self.assertIsInstance(ms_converted['a'], ms_Tensor)
    
    def test_edge_case_empty_tensor(self):
        """测试空张量边界情况"""
        empty_tensor = Tensor(np.array([], dtype=np.float32), self.env)
        self.assertEqual(empty_tensor.shape, (0,))
        self.assertEqual(empty_tensor.ndim, 1)

        empty_2d = Tensor(np.array([[], []], dtype=np.float32), self.env)
        self.assertEqual(empty_2d.shape, (2, 0))
    
    def test_edge_case_indexing(self):
        """测试索引边界情况"""
        tensor = Tensor(np.array([1, 2, 3], dtype=np.float32), self.env)
        self.assertEqual(tensor._elem[0].asnumpy(), 1.0)
        
        # 测试负索引
        self.assertEqual(tensor._elem[-1].asnumpy(), 3.0)
        
        # 越界索引异常处理
        with self.assertRaises(Exception):
            value = tensor._elem[10].asnumpy()
    
    def test_type_mismatch_error(self):
        """测试类型不匹配错误处理"""
        tensor = Tensor(np.array([1, 2, 3], dtype=np.float32), self.env)

        with self.assertRaises(TypeError):
            tensor.type_as("not_a_tensor")
    
    def test_invalid_operation_error(self):
        """测试无效操作的错误处理"""
        tensor = Tensor(np.array([1, 2, 3], dtype=np.float32), self.env)
        with self.assertRaises(Exception):
            # 传入一个会导致错误的函数
            tensor.apply_mindspore(lambda x: x + "invalid_type")
    
    def test_environment_edge_cases(self):
        """测试环境配置边界情况"""
        # 测试空配置
        empty_config = config.Configuration()
        empty_env = Environment(empty_config)
        self.assertIsNotNone(empty_env.config)
        
        # 测试属性覆盖的嵌套使用
        original_prng = self.env.param.prng
        
        with self.env.override_property(prng=100):
            self.assertEqual(self.env.param.prng, 100)

            with self.env.override_property(prng=200):
                self.assertEqual(self.env.param.prng, 200)
            self.assertEqual(self.env.param.prng, 100)
        
        # 恢复
        self.assertEqual(self.env.param.prng, original_prng)
    
    def test_error_handling_in_dispatch(self):
        """测试dispatch方法中的错误处理"""
        # 启用调试配置以测试异常处理
        self.env.config.debug_mixed_tensor = True
        
        try:
            # 尝试分发一个无效的操作，应该抛出OperatorNotFound异常
            with self.assertRaises(OperatorNotFound):
                self.env.dispatch("invalid_operation_name")
        finally:
            self.env.config.debug_mixed_tensor = False
    
    def test_boolean_scalar_tensor_conversion(self):
        """测试布尔标量张量转换边界情况"""
        bool_tensor = Tensor(np.array([True, False, True]), self.env)
        bool_values = bool_tensor.numpy().tolist()
        self.assertEqual(bool_values, [True, False, True])

        dtype_str = str(bool_tensor.dtype).lower()
        self.assertTrue('bool' in dtype_str or dtype_str == 'bool_')

        from mindspore import float32
        float_tensor = self.env._to_copy(bool_tensor, float32, None)
        self.assertEqual(float_tensor.dtype, float32)
        np.testing.assert_allclose(float_tensor.numpy(), np.array([1.0, 0.0, 1.0], dtype=np.float32))
    
    def test_large_tensor_handling(self):
        """测试大张量处理能力"""
        large_array = np.random.rand(1000, 1000)
        large_tensor = Tensor(large_array, self.env)
        self.assertEqual(large_tensor.shape, (1000, 1000))

        from mindspore import ops
        sum_result = large_tensor.apply_mindspore(ops.reduce_sum)
        self.assertIsInstance(sum_result, Tensor)


if __name__ == "__main__":
    unittest.main()