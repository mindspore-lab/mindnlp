import mindspore
input_ids = mindspore.Tensor(mindspore.ops.zeros((14,7)))
print(input_ids.shape)
input_ids_1 = input_ids.transpose(0,1)
print(input_ids_1.shape)