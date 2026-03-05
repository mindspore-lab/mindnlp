import mindtorch_v2 as torch


def test_custom_fwd_cast_inputs_to_bfloat16_in_autocast_region():
    class _Square(torch.autograd.Function):
        @staticmethod
        @torch.amp.custom_fwd(device_type="cpu", cast_inputs=torch.bfloat16)
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x

        @staticmethod
        @torch.amp.custom_bwd(device_type="cpu")
        def backward(ctx, grad_out):
            (x,) = ctx.saved_tensors
            return grad_out * (2 * x)

    x = torch.randn((2, 2), dtype=torch.float32)
    x.requires_grad = True

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        y = _Square.apply(x)

    assert y.dtype == torch.bfloat16


def test_custom_bwd_matches_forward_autocast_state():
    class _Square(torch.autograd.Function):
        @staticmethod
        @torch.amp.custom_fwd(device_type="cpu", cast_inputs=torch.bfloat16)
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x

        @staticmethod
        @torch.amp.custom_bwd(device_type="cpu")
        def backward(ctx, grad_out):
            (x,) = ctx.saved_tensors
            return grad_out * (2 * x)

    x = torch.randn((2, 2), dtype=torch.float32)
    x.requires_grad = True

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        y = _Square.apply(x)
        loss = y.sum()

    loss.backward()
    assert x.grad is not None
