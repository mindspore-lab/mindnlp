import torch
import torch.nn as nn

import torch4ms
from torch4ms.autograd.ms_autograd_function import extract_and_wrap_loss_fn


class SimpleRegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)


class SimpleClsNet(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x):
        return self.fc(x)


def _init_linear(module: nn.Module):
    with torch.no_grad():
        for p in module.parameters():
            p.fill_(0.5)


def _compare_loss(loss_name: str, model_ctor, loss_ctor, inputs, targets, atol=1e-5):
    print("=" * 60)
    print(f"Testing loss interception for {loss_name}")

    # Torch reference
    model_ref = model_ctor()
    _init_linear(model_ref)
    loss_ref_fn = loss_ctor()

    out_ref = model_ref(inputs)
    loss_ref = loss_ref_fn(out_ref, targets)
    loss_ref_val = float(loss_ref.item())
    print(f"[torch] loss = {loss_ref_val}")

    # torch4ms path
    model = model_ctor()
    _init_linear(model)
    loss_torch_fn = loss_ctor()

    env = torch4ms.default_env()
    with env:
        loss_wrapper = extract_and_wrap_loss_fn(model, loss_torch_fn, inputs, targets)
        loss_t4 = loss_wrapper.output

    # 转回 torch 标量比较数值
    loss_t4_val = float(loss_t4.torch().item())
    print(f"[torch4ms] loss = {loss_t4_val}")

    diff = abs(loss_t4_val - loss_ref_val)
    print(f"[compare] |diff| = {diff}")
    if diff < atol:
        print(f"[OK] {loss_name} forward loss matches within atol={atol}")
    else:
        print(f"[WARN] {loss_name} forward loss mismatch (>|{atol}|)")


def main():
    # 公共输入
    x_reg = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]], dtype=torch.float32)
    y_reg = torch.tensor([[10.0],
                          [20.0]], dtype=torch.float32)

    y_bce = torch.tensor([[0.0],
                          [1.0]], dtype=torch.float32)

    # 回归类损失
    _compare_loss("MSELoss",
                  SimpleRegNet,
                  lambda: nn.MSELoss(reduction="mean"),
                  x_reg,
                  y_reg)

    _compare_loss("L1Loss",
                  SimpleRegNet,
                  lambda: nn.L1Loss(reduction="mean"),
                  x_reg,
                  y_reg)

    _compare_loss("SmoothL1Loss",
                  SimpleRegNet,
                  lambda: nn.SmoothL1Loss(beta=1.0, reduction="mean"),
                  x_reg,
                  y_reg)

    _compare_loss("HuberLoss",
                  SimpleRegNet,
                  lambda: nn.HuberLoss(delta=1.0, reduction="mean"),
                  x_reg,
                  y_reg)

    # 二分类 BCEWithLogits
    _compare_loss("BCEWithLogitsLoss",
                  SimpleRegNet,
                  lambda: nn.BCEWithLogitsLoss(reduction="mean"),
                  x_reg,
                  y_bce)


if __name__ == "__main__":
    main()

