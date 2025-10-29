import torch4ms
import torch
from torch4ms import MSDispatchMode, MSFunctionMode
import mindspore

dispatch_mode = MSDispatchMode()
function_mode = MSFunctionMode()

dispatch_mode.__enter__()
function_mode.__enter__()

def test_add():
    x = torch.tensor(1)
    y = torch.tensor(2)
    z = x + y
    print(z)
