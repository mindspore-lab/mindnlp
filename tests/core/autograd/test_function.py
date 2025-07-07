import mindnlp
from mindnlp import core as torch
from mindnlp.core.autograd import Function

class Test(Function):

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x + y + 1
    
    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.saved_tensors
        print(x, y)
        return torch.ones_like(x), torch.zeros_like(y)

def fn_test(x, y):
    return Test.apply(x, y)

def test_function_no_record_forward_inputs():
    x = torch.randn(3, 3, requires_grad=True)
    y = torch.randn(3, requires_grad=True)
    out = fn_test(x, y)
    out.backward()
    print(x.requires_grad)
    print(y.requires_grad)
