## 一、特性概述

### 1.2 场景分析

最初的词向量是one-hot形式的向量，即只有单词所在的那一维是1，其他维都是0，向量长度和词汇表大小一样。
缺点显而易见，容易造成维度灾难，并且对词语之间的语义关系起不到任何表达作用。所以考虑用稠密的实数向量来对词语进行表示。
如果用稠密实数向量来表示词语，这样可以被计算机理解，如果能找到比较好的映射方法，那么能够很好地表示语义相关性。
embedding用一个数值向量表示一个对象，主要用以表示对象之间的关系。

![image.png](https://miro.medium.com/max/1400/1*sXNXYfAqfLUeiDXPCo130w.png)

### 1.3 特性影响分析

表 1-1 ：约束说明

| *支持后端*                 | *支持模式*              | 支持平台          |
| ---------------------------- | ------------------------- | ----------------- |
| *ASCEND/GPU/**CPU*** | *静态图模式/动态图模式* | WINDOWS/MAC/LINUX |


## 二、详细设计

### 2.1总体方案描述

### 2.2 基本功能设计

from_pretrained:

```python
def auto_mixed_precision(network, amp_level="O0"):
    if amp_level == "O0":
        pass
    elif amp_level == "O1":
        _auto_white_list(network)
    elif amp_level == "O2":
        _auto_black_list(network)
    elif amp_level == "O3":
        network.to_float(mstype.float16)
    else:
        raise ValueError("The amp level {} is not supported".format(amp_level))
```

样例：network = Net()

```python
new_network = auto_mixed_precision(network, "O1")
```

内部实现：

LossScaler基类：

```python
@ms_class
class LossScaler():
    def scale(self, inputs):
        ...
    def unscale(self, inputs):
        ...
    def adjust(self, grads_finite):
        ...
```

代码样例：

```python
net = Net()
loss_fn = nn.BCELoss(reduction='mean')
opt = nn.Adam(generator.trainable_params(), learning_rate=0.01)
loss_scaler = amp.DynamicLossScaler(scale_value=2**10, scale_factor=2, scale_window=50)

def net_forward(data, label):
    out = net(data)
    loss_value = loss_fn(out, label)
    scaled_loss = loss_scaler.scale(loss_value)
    return scaled_loss, out

grad_fn = ops.value_and_grad(net_forward, None, net.trainable_params())

@ms_function
def train_step(x, y):
    (loss_value, logits), grads = grad_fn(x, y)
    loss_value = loss_scaler.unscale(loss_value)

    is_finite = all_finite(grads)
    if is_finite:
        grads = loss_scaler.unscale(grads)
        loss_value = ops.depend(loss_value, opt(grads))
    loss_scaler.adjust(is_finite)
    return loss_value

for epoch in epochs:
    for data, label in datasets:
        loss = train_step(data, label)

```


## 三、对外接口

*接口包含直接的对外接口、环境变量、配置文件等。*

_1、接口说明（函数输入输出参数/属性/返回值）_

| 函数名 | 输入参数 | 返回值 |
| ---- | ---- | ---- |
| mindnlp.dataset.IMDB | root:str\split\xxx | List(datasets)/ dataset |


### 遗留问题：
