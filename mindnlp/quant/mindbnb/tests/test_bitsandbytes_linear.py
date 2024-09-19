import torch
import torch.nn as nn
import numpy as np

from bitsandbytes.nn import Linear8bitLt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)

# 创建模型
int8_model = Linear8bitLt(2, 4, has_fp16_weights=False)

# 初始化权重和偏置
weight = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float16)
bias = torch.tensor([1, 2], dtype=torch.float16)

# 设置模型的权重和偏置
int8_model.weight.data = weight
int8_model.bias.data = bias

int8_model.to(device)

# 输入数据
input_data = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float16).to(device)

# 模型输出
int8_output = int8_model(input_data)

print(int8_output)
