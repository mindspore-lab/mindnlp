
import numpy as np
import time
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds

from mindspore import context, Tensor
from mindspore.communication import init, get_rank

# context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU") # mode=context.PYNATIVE_MODE,
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU") # mode=context.PYNATIVE_MODE,


mindspore.set_seed(0)
# init('nccl')

class MLP(nn.Cell):
    """Tset MLP."""
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
        # return self.layers.construct(X)
        # SequentialCell have bugs: construct not dynamic
        for cell in self.layers:
            X = cell(X)
        # return self.layers(X)
        return X
    

class MyIterable:
    """Custom datasets."""
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
X = np.random.rand(1000, 20).astype(np.float32)
# X = mindspore.ops.rand((1000,20))
y = (X.sum(1) > 10).astype(np.int32)

n_train = 800
batch_size = 64
max_epochs = 50
lr = 2e-3

train_ds = ds.GeneratorDataset(
    source=MyIterable(X[:n_train], y[:n_train]),
    column_names=["data", "label"],
    shuffle=True
)
eval_ds = ds.GeneratorDataset(
    source=MyIterable(X[n_train:], y[n_train:]),
    column_names=["data", "label"],
)

train_dataloader = train_ds.batch(batch_size=batch_size)
eval_dataloader = eval_ds.batch(batch_size=batch_size)

mlp = MLP()
optimizer = nn.Adam(mlp.trainable_params(), learning_rate=lr)
criterion = nn.CrossEntropyLoss()
# # print_net_params(mlp)


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
    
# train(mlp, optimizer, criterion, train_dataloader, eval_dataloader, epochs=max_epochs)
train(mlp, optimizer, criterion, train_dataloader, eval_dataloader, epochs=max_epochs)