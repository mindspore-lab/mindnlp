

def test_AllReduce_mean():
    import numpy as np
    from mindspore.communication import init, get_rank, get_group_size
    import mindspore as ms
    import mindspore.nn as nn
    import mindspore.ops as ops

    init()
    rank_size = get_group_size()
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.all_reduce_sum = ops.AllReduce(ops.ReduceOp.SUM)

        def construct(self, x):
            new_grads_mean = self.all_reduce_sum(x) / rank_size
            new_grad = new_grads_mean
            return new_grad

    rank_id_value = get_rank() # Current NPU number 0,...,7
    print('rank_id_value=',rank_id_value)
    input_x = ms.Tensor(np.array([[rank_id_value]]).astype(np.float32))
    print('input_x=',input_x)
    net = Net()
    output = net(input_x)
    print("mean:",output) # sum(0, rank_size) / rank_size




if __name__ == '__main__':
    test_AllReduce_mean()

