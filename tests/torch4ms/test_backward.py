"""
测试 make_train_step 功能（使用固定输入，验证梯度）
"""
import torch
import torch.nn as nn
import torch4ms
from torch4ms.autograd.ms_autograd_function import extract_and_wrap_loss_fn


def test_train_step_fixed():
    """对比 torch4ms 与原生 torch 的 loss/grad 结果（固定输入）"""
    print("=" * 60)
    print("测试训练步骤（固定输入）")
    print("=" * 60)

    # 固定输入/标签（两边复用，便于对比）
    inputs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    labels = torch.tensor([[10.0], [20.0]], dtype=torch.float32)

    print(f"Inputs: {inputs}")
    print(f"Labels: {labels}")

    # -------------------------
    # 1) torch4ms 路径（MindSpore GradOperation）
    # -------------------------
    env = torch4ms.default_env()
    with env:
        model = nn.Linear(3, 1)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(0.0)

        from mindspore import ops

        def mse_loss_fn(output, target):
            """手动实现的 MSE loss（MindSpore 语义）"""
            return ops.reduce_mean((output - target) ** 2)

        loss_wrapper = extract_and_wrap_loss_fn(model, mse_loss_fn, inputs, labels)
        loss_t4 = loss_wrapper.output
        print(f"[torch4ms] Model weight: {model.weight}")
        print(f"[torch4ms] Model bias: {model.bias}")
        print(f"[torch4ms] Loss: {loss_t4}")

        loss_t4.backward()

        # 将 torch4ms 的梯度映射回 torch 参数，便于打印/对比
        from torch4ms.ops import mappings

        param_tensors = loss_wrapper.inputs
        for i, (_, param) in enumerate(model.named_parameters()):
            if i < len(param_tensors) and param_tensors[i].grad is not None:
                param.grad = mappings.ms2t(param_tensors[i].grad._elem)

        t4_w_grad = model.weight.grad.detach().clone() if model.weight.grad is not None else None
        t4_b_grad = model.bias.grad.detach().clone() if model.bias.grad is not None else None

    print(f"[torch4ms] weight.grad: {t4_w_grad}")
    print(f"[torch4ms] bias.grad: {t4_b_grad}")

    # -------------------------
    # 2) 原生 torch 路径（PyTorch autograd）
    # -------------------------
    model_ref = nn.Linear(3, 1)
    with torch.no_grad():
        model_ref.weight.fill_(1.0)
        model_ref.bias.fill_(0.0)

    out_ref = model_ref(inputs)
    loss_ref = torch.mean((out_ref - labels) ** 2)
    loss_ref.backward()

    ref_w_grad = model_ref.weight.grad.detach().clone()
    ref_b_grad = model_ref.bias.grad.detach().clone()

    print(f"[torch] Loss: {loss_ref.item()}")
    print(f"[torch] weight.grad: {ref_w_grad}")
    print(f"[torch] bias.grad: {ref_b_grad}")

    # -------------------------
    # 对比
    # -------------------------
    if t4_w_grad is None or t4_b_grad is None:
        print("[COMPARE] torch4ms 梯度为空，无法对比")
        return

    w_diff = (t4_w_grad - ref_w_grad).abs().max().item()
    b_diff = (t4_b_grad - ref_b_grad).abs().max().item()
    print(f"[COMPARE] max|weight.grad diff| = {w_diff}")
    print(f"[COMPARE] max|bias.grad diff|   = {b_diff}")

    if w_diff < 1e-6 and b_diff < 1e-6:
        print("OK: torch4ms 与 torch 的梯度一致")
    else:
        print("WARNING: torch4ms 与 torch 的梯度存在差异（请检查算子/广播/精度）")


if __name__ == "__main__":
    try:
        test_train_step_fixed()
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
