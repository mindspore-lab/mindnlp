import torch4ms 
import torch 
import mindspore 
import numpy as np

env = torch4ms.default_env()
env.__enter__() 

def test_matrix_operations():
    """测试矩阵乘法和加法组合运算"""
    # 创建测试数据
    np_x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np_y = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    np_z = np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32)

    x = torch.tensor(np_x)
    y = torch.tensor(np_y)
    z = torch.tensor(np_z)
    result = torch.matmul(x, y) + z

    print(f"x = {x}")
    print(f"y = {y}")
    print(f"z = {z}")
    print(f"x * y + z = {result}")

    expected = np.matmul(np_x, np_y) + np_z
    print(f"\n预期结果:")
    print(f"{expected}")

    np_result = result.detach().numpy()
    print(f"\n数值验证:")
    print(f"结果是否接近预期: {np.allclose(np_result, expected, atol=1e-5)}")

def test_activation_functions():
    """测试激活函数"""
    np_data = np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]], dtype=np.float32)
    x = torch.tensor(np_data)
    
    print("\n" + "="*40)
    print("测试激活函数:")
    print(f"输入: {x}")

    relu_result = torch.relu(x)
    print(f"\nReLU结果: {relu_result}")

    sigmoid_result = torch.sigmoid(x)
    print(f"Sigmoid结果: {sigmoid_result}")

    tanh_result = torch.tanh(x)
    print(f"Tanh结果: {tanh_result}")

if __name__ == "__main__":
    print("PyTorch版本: {}".format(torch.__version__))
    print("MindSpore版本: {}".format(mindspore.__version__))
    print("="*40)
    
    test_matrix_operations()
    test_activation_functions()
    env.__exit__(None, None, None)