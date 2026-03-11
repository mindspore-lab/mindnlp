import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import math
import torch
import torch.nn as nn
import torch.optim as optim

import torch4ms
from torch4ms.autograd.ms_autograd_function import extract_and_wrap_loss_fn
from torch4ms.optim import Torch4msOptimizer

NUM_EPOCHS = 2
LR = 0.01
ATOL_REL = 0.2


def get_dataloader():
    torch.manual_seed(0)
    # [batch, seq, in_dim]
    x = torch.randn(4, 8, 16, dtype=torch.float32)
    y = torch.randn(4, 1, dtype=torch.float32)
    return [(x, y)]


class TinyTransformerRegressor(nn.Module):
    def __init__(self, in_dim=16, d_model=32, nhead=4, ff_dim=64, num_layers=2):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.0,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.encoder(x)
        x = torch.mean(x, dim=1)
        x = self.head(x)
        return x


def init_model(model: nn.Module):
    torch.manual_seed(42)
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.1, 0.1)


def _grad_norms(model: nn.Module):
    norms = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            norms[name] = None
        else:
            norms[name] = float(p.grad.norm().item())
    return norms


def _print_grad_norms(prefix: str, norms):
    for name, v in norms.items():
        if v is None:
            print(f"{prefix} {name}.grad = None")
        else:
            print(f"{prefix} {name}.grad norm = {v:.8f}")


def _is_valid(v):
    return (v is not None) and (not math.isnan(v)) and (not math.isinf(v))


def _status_pair(t4, th):
    if not _is_valid(t4):
        return "BAD_INVALID"
    if th is None:
        return "UNKNOWN"
    if abs(th) > 1e-6 and abs(t4) < 1e-8:
        return "MISMATCH_ZERO_GRAD"
    rel = abs(t4 - th) / max(abs(th), 1e-12)
    return "OK" if rel < ATOL_REL else "WARN_DIFF"


def train_with_torch4ms():
    print("=== torch4ms TinyTransformer ===")
    dataloader = get_dataloader()
    criterion = nn.MSELoss()

    model = TinyTransformerRegressor()
    init_model(model)
    optimizer = Torch4msOptimizer(optim.SGD(model.parameters(), lr=LR), model)

    last_norms = {}
    env = torch4ms.default_env()
    with env:
        for epoch in range(NUM_EPOCHS):
            for step, (inputs, labels) in enumerate(dataloader):
                print(f"[torch4ms] epoch={epoch}, step={step}")
                loss_wrapper = extract_and_wrap_loss_fn(model, criterion, inputs, labels)
                loss_t4 = loss_wrapper.output
                print(f"[torch4ms] loss = {loss_t4}")

                loss_t4.backward(module=model)
                last_norms = _grad_norms(model)
                _print_grad_norms("[torch4ms]", last_norms)

                optimizer.step()
                optimizer.zero_grad()
    return last_norms


def train_with_torch():
    print("=== torch TinyTransformer ===")
    dataloader = get_dataloader()
    criterion = nn.MSELoss()

    model = TinyTransformerRegressor()
    init_model(model)
    optimizer = optim.SGD(model.parameters(), lr=LR)

    last_norms = {}
    for epoch in range(NUM_EPOCHS):
        for step, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            print(f"[torch] epoch={epoch}, step={step}, loss = {loss.item()}")
            loss.backward()
            last_norms = _grad_norms(model)
            _print_grad_norms("[torch]", last_norms)
            optimizer.step()
    return last_norms


def main():
    t4_norms = train_with_torch4ms()
    print()
    th_norms = train_with_torch()

    print()
    print("=== transformer stability summary ===")
    all_names = sorted(set(t4_norms.keys()) | set(th_norms.keys()))
    for name in all_names:
        t4 = t4_norms.get(name)
        th = th_norms.get(name)
        status = _status_pair(t4, th)
        print(f"[summary] {name}: torch4ms={t4}, torch={th}, status={status}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        print(f"[ERROR] {exc}")
        traceback.print_exc()
