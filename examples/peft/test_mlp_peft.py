
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
lr = 0.002

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
        X = self.layers(X)

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
    target_modules=["layers.0", "layers.2"],
    modules_to_save=["layers.4"],
)
import copy
model = MLP(2000)
module_copy = copy.deepcopy(model) 
peft_model = peft.get_peft_model(model, peft_config=config)

print_net_params(peft_model)
print(peft_model)

optimizer = nn.Adam(peft_model.trainable_params(), learning_rate=lr)
criterion = nn.CrossEntropyLoss()

train(peft_model, optimizer, criterion, train_dataloader, eval_dataloader, epochs=max_epochs)

params_before = dict(module_copy.parameters_and_names())
print(params_before.keys())
for name, param in peft_model.base_model.parameters_and_names():
    if "lora" in name:
        continue
    name_before = name.partition(".")[-1].replace("original_", "").replace("module.", "").replace("modules_to_save.default.", "")
    param_before = params_before[name_before]
    
    if np.allclose(param.asnumpy(), param_before.asnumpy()):
        print(f"Parameter {name_before:<13} | {param.numel():>7} parameters | not updated")
    else:
        print(f"Parameter {name_before:<13} | {param.numel():>7} parameters | updated")
