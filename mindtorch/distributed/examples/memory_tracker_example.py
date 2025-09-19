# mypy: allow-untyped-defs
import mindtorchvision

import mindtorch
from mindtorch.distributed._tools import MemoryTracker


def run_one_model(net: mindtorch.nn.Module, input: mindtorch.Tensor):
    net.cuda()
    input = input.cuda()

    # Create the memory Tracker
    mem_tracker = MemoryTracker()
    # start_monitor before the training iteration starts
    mem_tracker.start_monitor(net)

    # run one training iteration
    net.zero_grad(True)
    loss = net(input)
    if isinstance(loss, dict):
        loss = loss["out"]
    loss.sum().backward()
    net.zero_grad(set_to_none=True)

    # stop monitoring after the training iteration ends
    mem_tracker.stop()
    # print the memory stats summary
    mem_tracker.summary()
    # plot the memory traces at operator level
    mem_tracker.show_traces()


run_one_model(torchvision.models.resnet34(), mindtorch.rand(32, 3, 224, 224, device="cuda"))
