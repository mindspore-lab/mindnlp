
import time
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
import mindnlp.peft as peft

from mindspore import context, Tensor
from mindspore.communication import init, get_rank

# GRAPH_MODE have bugs
# context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU") # mode=context.PYNATIVE_MODE,
mindspore.set_seed(0)
# init('nccl')

# mindspore 构造一个测试数据集
X = np.random.rand(1000, 20).astype(np.float32)
y = (X.sum(1) > 10).astype(np.int32)

n_train = 800
batch_size = 64
max_epochs = 50
lr = 2e-4

class RandomAccessDataset:
    def __init__(self, X, y):
        self._index = 0
        self._data = X
        self._label = y

    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)

train_ds = ds.GeneratorDataset(
    source=RandomAccessDataset(X[:n_train], y[:n_train]),
    column_names=["data", "label"],
    shuffle=True
)
eval_ds = ds.GeneratorDataset(
    source=RandomAccessDataset(X[n_train:], y[n_train:]),
    column_names=["data", "label"],
)

train_dataloader = train_ds.batch(batch_size=batch_size)
eval_dataloader = eval_ds.batch(batch_size=batch_size)

class MLP(nn.Cell):
    def __init__(self, hidden: int = 2000):
        super().__init__()
        self.layers = nn.SequentialCell(
            nn.Dense(20, hidden),
            nn.ReLU(),
            nn.Dense(hidden, hidden),
            nn.ReLU(),
            nn.Dense(hidden, 2),
            nn.LogSoftmax(axis=-1)
        )

    def construct(self, X):
        # SequentialCell have bugs: construct not dynamic
        # X = self.layers(X)

        X = self.layers[0](X)

        X = self.layers[1](X)

        X = self.layers[2](X)

        X = self.layers[3](X)

        X = self.layers[4](X)

        X = self.layers[5](X)

        return X

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
            # print(xb, yb)
            # print("====loss")
            # print(loss)
            # print("=====grad")
            # print(grad)
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
    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch()
        eval_loss = eval_one_epoch()

        if epoch % 2 == 0:
            print(f"epoch:{epoch}  train_loss:{train_loss}  eval_loss:{eval_loss}")
    

config = peft.LoraConfig(
    r=8,
    target_modules=["layers.0", "layers.2", "layers.4"],
    modules_to_save=[],
)

mlp = MLP(2000)
peft_mlp = peft.get_peft_model(mlp, peft_config=config)

print_net_params(peft_mlp)
print(peft_mlp)
print(peft_mlp.trainable_params())

optimizer = nn.Adam(peft_mlp.trainable_params(), learning_rate=lr)
criterion = nn.CrossEntropyLoss()
# peft_mlp.print_trainable_parameters()

# # print(peft_mlp.base_model.model)
# # print(peft_mlp)
# # print_net_params(peft_mlp.base_model.model)
# # print_net_params(peft_mlp)
# # optimizer = torch.optim.Adam(peft_mlp.parameters(), lr=lr)


train(peft_mlp, optimizer, criterion, train_dataloader, eval_dataloader, epochs=max_epochs)


