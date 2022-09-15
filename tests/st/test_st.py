# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test LeNet5 network for training in GPU."""

import pytest
from mindspore import ops
from mindspore.common.initializer import Normal
from mindspore import nn
from mindspore import dataset as ds
import mindspore as ms
import numpy as np
import mindtext

print(mindtext)

def get_data(num, w=2.0, b=3.0):
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)


def create_dataset(num_data, batch_size=16, repeat_size=1):
    input_data = ds.GeneratorDataset(
        list(get_data(num_data)), column_names=['data', 'label'])
    input_data = input_data.batch(batch_size, drop_remainder=True)
    input_data = input_data.repeat(repeat_size)
    return input_data


ms.set_context()
ds_train = create_dataset(
    1600, batch_size=16, repeat_size=1)
print("The dataset size of ds_train:", ds_train.get_dataset_size())
step_size = ds_train.get_dataset_size()
dict_datasets = next(ds_train.create_dict_iterator())


class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        fx = self.fc(x)
        return fx


net = LinearNet()  # 初始化线性回归网络


class MyMAELoss(nn.LossBase):
    """定义损失"""

    def __init__(self):
        super(MyMAELoss, self).__init__()
        self.abs = ops.Abs()

    def construct(self, predict, target):
        x = self.abs(target - predict)
        return self.get_loss(x)


class MyMomentum(nn.Optimizer):
    """定义优化器"""

    def __init__(self, params, learning_rate, momentum=0.9):
        super(MyMomentum, self).__init__(learning_rate, params)
        self.moment = ms.Parameter(
            ms.Tensor(momentum, ms.float32), name="moment")
        self.momentum = self.parameters.clone(prefix="momentum", init="zeros")
        self.assign = ops.Assign()

    def construct(self, gradients):
        """construct输入为梯度，在训练中自动传入梯度gradients"""
        lr = self.get_lr()
        params = self.parameters  # 待更新的权重参数
        for i in range(len(params)):
            self.assign(self.momentum[i], self.momentum[i]
                        * self.moment + gradients[i])
            update = params[i] - self.momentum[i] * lr  # 带有动量的SGD算法
            self.assign(params[i], update)
        return params


class MyWithLossCell(nn.Cell):
    """定义损失网络"""

    def __init__(self, backbone, loss_fn):
        """实例化时传入前向网络和损失函数作为参数"""
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, label):
        """连接前向网络和损失函数"""
        out = self.backbone(data)
        return self.loss_fn(out, label)

    def backbone_network(self):
        """要封装的骨干网络"""
        return self.backbone


# ### 定义训练流程
class MyTrainStep(nn.TrainOneStepCell):
    """定义训练流程"""

    def __init__(self, network, optimizer):
        """参数初始化"""
        super(MyTrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, data, label):
        """构建训练过程"""
        weights = self.weights
        loss = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        return loss, self.optimizer(grads)


class MyMAE(nn.Metric):
    """定义metric"""

    def __init__(self):
        super(MyMAE, self).__init__()
        self.clear()

    def clear(self):
        """初始化变量abs_error_sum和samples_num"""
        self.abs_error_sum = 0
        self.samples_num = 0

    def update(self, *inputs):
        """更新abs_error_sum和samples_num"""
        y_pred = inputs[0].asnumpy()
        y = inputs[1].asnumpy()

        # 计算预测值与真实值的绝对误差
        error_abs = np.abs(y.reshape(y_pred.shape) - y_pred)
        self.abs_error_sum += error_abs.sum()
        self.samples_num += y.shape[0]  # 样本的总数

    def eval(self):
        """计算最终评估结果"""
        return self.abs_error_sum / self.samples_num


# ## 自定义验证流程
class MyWithEvalCell(nn.Cell):
    """定义验证流程"""

    def __init__(self, network):
        super(MyWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, data, label):
        outputs = self.network(data)
        return outputs, label


@pytest.mark.level0
@pytest.mark.env_single
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
def test_linear_fitting():
    """
    Feature: linear fitting network.
    Description: test linear fitting network for training.
    Expectation: success.
    """
    loss_func = MyMAELoss()                         # 损失函数
    opt = MyMomentum(net.trainable_params(), 0.01)  # 优化器

    net_with_criterion = MyWithLossCell(net, loss_func)  # 构建损失网络
    train_net = MyTrainStep(net_with_criterion, opt)     # 构建训练网络

    for data in ds_train.create_dict_iterator():
        train_net(data['data'], data['label'])                  # 执行训练，并更新权重

    # 执行推理并评估：
    ds_eval = create_dataset(
        160, batch_size=16, repeat_size=1)

    eval_net = MyWithEvalCell(net)  # 定义评估网络
    eval_net.set_train(False)
    mae = MyMAE()

    # 执行推理过程
    for data in ds_eval.create_dict_iterator():
        output, eval_y = eval_net(data['data'], data['label'])
        mae.update(output, eval_y)

    mae_result = mae.eval()
    assert mae_result > 0.5
