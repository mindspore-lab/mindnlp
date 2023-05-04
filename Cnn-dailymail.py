import mindspore.context as context
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore.common.initializer import Normal
from mindspore.common.parameter import Parameter
import numpy as np

import mindflow
from mindflow.ops import fluid
from mindflow import Model
from mindflow.dataset import Dataset, DataLoader

# 设置上下文环境为GPU
context.set_context(device_target="GPU")

# 定义泰勒-格林涡流动模型
class TaylorGreenNet(nn.Cell):
    def __init__(self, num_classes=1):
        super(TaylorGreenNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, pad_mode='same', has_bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad_mode='same', has_bias=True)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(64 * 128 * 128, num_classes, weight_init=Normal(0.01))

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# 定义输入数据和标签
train_x_data = np.load("train_x_data.npy")
train_y_data = np.load("train_y_data.npy")

# 定义MindFlow数据集和数据加载器
train_dataset = Dataset(train_x_data, train_y_data)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 定义MindFlow模型
net = TaylorGreenNet(num_classes=1)
loss_fn = nn.MSELoss()
opt = nn.Adam(params=net.trainable_params(), learning_rate=0.001)
model = Model(net, loss_fn, opt, metrics={"MSE": nn.MSELoss()})

# 训练模型
history = model.fit(train_loader, epochs=100, verbose=1)

# 展示训练结果
import matplotlib.pyplot as plt

plt.plot(history["loss"], label="train loss")
plt.plot(history["MSE"], label="train MSE")
plt.legend()
plt.show()
