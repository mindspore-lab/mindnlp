import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import torch.nn as nn
import torch.optim as optim
import math

import torch4ms
from torch4ms.autograd.ms_autograd_function import extract_and_wrap_loss_fn
from torch4ms.optim import Torch4msOptimizer

NUM_EPOCHS = 2
LR = 0.01


def get_dataloader():
    torch.manual_seed(0)
    x = torch.randn(4, 1, 8, 8, dtype=torch.float32)
    y = torch.randn(4, 1, dtype=torch.float32)
    return [(x, y)]


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(4 * 8 * 8, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(1)
        x = self.drop(x)
        x = self.fc(x)
        return x


def init_model(model: nn.Module):
    torch.manual_seed(42)
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.1, 0.1)


def _print_grad_stats(prefix: str, model: nn.Module):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"{prefix} {name}.grad = None")
        else:
            print(f"{prefix} {name}.grad norm = {param.grad.norm().item():.8f}")


def _collect_grad_norms(model: nn.Module):
    result = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            result[name] = None
        else:
            result[name] = float(param.grad.norm().item())
    return result


def _is_valid_norm(v):
    return (v is not None) and (not math.isnan(v)) and (not math.isinf(v))


def _status_by_pair(torch4ms_v, torch_v):
    if not _is_valid_norm(torch4ms_v):
        return "BAD_INVALID"
    if torch_v is None:
        return "UNKNOWN"
    # torch 明显有梯度而 torch4ms 几乎为 0，判为不一致
    if abs(torch_v) > 1e-6 and abs(torch4ms_v) < 1e-8:
        return "MISMATCH_ZERO_GRAD"
    # 二者都非零时给一个粗粒度相对差异判断
    denom = max(abs(torch_v), 1e-12)
    rel_diff = abs(torch4ms_v - torch_v) / denom
    if rel_diff < 0.15:
        return "OK"
    return "WARN_DIFF"


def train_with_torch4ms_cnn():
    print("=== torch4ms CNN path ===")
    dataloader = get_dataloader()
    criterion = nn.MSELoss()

    model = SimpleCNN()
    init_model(model)
    optimizer = Torch4msOptimizer(optim.SGD(model.parameters(), lr=LR), model)

    env = torch4ms.default_env()
    last_grad_norms = {}
    with env:
        for epoch in range(NUM_EPOCHS):
            for step, (inputs, labels) in enumerate(dataloader):
                print(f"[torch4ms] epoch={epoch}, step={step}")
                loss_wrapper = extract_and_wrap_loss_fn(model, criterion, inputs, labels)
                loss_t4 = loss_wrapper.output
                print(f"[torch4ms] loss = {loss_t4}")

                loss_t4.backward(module=model)
                _print_grad_stats("[torch4ms]", model)
                last_grad_norms = _collect_grad_norms(model)

                optimizer.step()
                optimizer.zero_grad()

    return last_grad_norms


def train_with_torch_cnn():
    print("=== torch CNN ===")
    dataloader = get_dataloader()
    criterion = nn.MSELoss()

    model = SimpleCNN()
    init_model(model)
    optimizer = optim.SGD(model.parameters(), lr=LR)

    last_grad_norms = {}
    for epoch in range(NUM_EPOCHS):
        for step, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            print(f"[torch] epoch={epoch}, step={step}, loss = {loss.item()}")
            loss.backward()
            _print_grad_stats("[torch]", model)
            last_grad_norms = _collect_grad_norms(model)
            optimizer.step()

    return last_grad_norms


def main():
    t4_norms = train_with_torch4ms_cnn()
    print()
    torch_norms = train_with_torch_cnn()

    print()
    print("=== stability regression summary ===")
    for name in sorted(set(t4_norms.keys()) | set(torch_norms.keys())):
        t4_v = t4_norms.get(name)
        th_v = torch_norms.get(name)
        status = _status_by_pair(t4_v, th_v)
        print(f"[summary] {name}: torch4ms={t4_v}, torch={th_v}, torch4ms_status={status}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        print(f"[ERROR] {exc}")
        traceback.print_exc()
