
import numpy as np
import time
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds

from mindspore import context, Tensor
from mindspore.communication import init, get_rank



context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
# init('nccl')

class MLP(nn.Cell):
    def __init__(self, hidden=2000):
        super(MLP, self).__init__()
        self.layers = nn.SequentialCell(
            nn.Dense(20, hidden),
            nn.ReLU(),
            nn.Dense(hidden, hidden),
            nn.ReLU(),
            nn.Dense(hidden, 2),
            nn.LogSoftmax(axis=-1)
        )

    def construct(self, X):
        return self.layers(X)
    


class MyIterable:
    def __init__(self, X: Tensor, y:Tensor):
        self._index = 0
        self._data = X
        self._label = y

    def __next__(self):
        if self._index >= len(self._data):
            raise StopIteration
        else:
            item = (self._data[self._index], self._label[self._index])
            self._index += 1
            return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self._data)
    

def print_net_params(net: nn.Cell):
    "print all parameters in net."
    print("print all params")
    all_parameter = []
    for item in net.get_parameters():
        all_parameter.append(item)
        print(item.name, item.data.shape)
    print(f"all parameter numbers: {len(all_parameter)}")

    # Obtain trainable parameters.
    trainable_params = net.trainable_params()
    for item in trainable_params:
        print(item.name, item.data.shape)
    print(f"trainable parameter numbers: {len(trainable_params)}")


# mindspore 构造一个测试数据集
X = mindspore.ops.rand((1000,20), seed=0)
y = (X.sum(1) > 10).int()

n_train = 800
batch_size = 64

train_ds = ds.GeneratorDataset(
    source=MyIterable(X[:n_train], y[:n_train]),
    column_names=["data", "label"],
    shuffle=True
)
eval_ds = ds.GeneratorDataset(
    source=MyIterable(X[n_train:], y[n_train:]),
    column_names=["data", "label"],
)

train_dataloader = train_ds.batch(batch_size=batch_size, drop_remainder=True)
eval_dataloader = eval_ds.batch(batch_size=batch_size, drop_remainder=True)

lr = 0.002
batch_size = 64
max_epochs = 50

mlp = MLP()

print_net_params(mlp)

optimizer = nn.Adam(mlp.trainable_params(), learning_rate=lr)
criterion = nn.CrossEntropyLoss()





def train(model, optimizer, criterion, train_dataloader, eval_dataloader, epochs):
    def forward_fn(data, label):
        logits = model(data)
        loss = criterion(logits, label)
        return loss, logits

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_one_epoch():
        model.set_train()
        total_loss = 0

        for xb, yb in train_dataloader:
            # forward + compute grad
            (loss, logits), grad = grad_fn(xb, yb)
            # update model params
            optimizer(grad)
            total_loss += loss
        
        return total_loss / len(train_dataloader)
    
    def eval_one_epoch():
        model.set_train(False)
        total_loss = 0

        for xb, yb in eval_dataloader:
            (loss, logits), grad = grad_fn(xb, yb)
            total_loss += loss

        return total_loss / len(eval_dataloader)

    # train start from here
    for epoch in range(epochs):
        train_loss = train_one_epoch()
        eval_loss = eval_one_epoch()

        if epoch % 2 == 0:
            print(f"epoch:{epoch}  train_loss:{train_loss}  eval_loss:{eval_loss}")
    



# %time train(mlp, optimizer, criterion, train_dataloader, eval_dataloader, epochs=max_epochs)


# check
trainable_params = mlp.trainable_params()
print(mlp)
trainable_params


import mindnlp.peft as peft


# target_modules are modules to add PEFT params
# modules_to_save are original modules, not freezed.
config = peft.LoraConfig(
    r=8,
    target_modules=["layers.0", "layers.2"],
    modules_to_save=["layers.4"],
)


config


mlp = MLP()
peft_mlp = peft.get_peft_model(mlp, peft_config=config)

print(peft_mlp)

peft_mlp.print_trainable_parameters()
# print_net_params(peft_mlp)




