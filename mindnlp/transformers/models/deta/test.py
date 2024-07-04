import torch
from mindspore import ops
import mindspore as ms

idx = ms.Tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])


a = torch.Tensor([[0.31439176, 0.6003434, 0.364401, 0.6503287]])
b = torch.Tensor([0.26844662])
# print(torch.stack((a, b)))

print(torch.ops.torchvision.nms)

for class_id in ops.unique(idx)[0]:
    curr_indices = ops.nonzero(idx == class_id)
    print(curr_indices[:, 0])


for class_id in torch.unique(idxs):
    curr_indices = torch.where(idxs == class_id)[0]
    print(curr_indices)
