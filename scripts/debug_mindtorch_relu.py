import mindtorch_v2 as torch


def main():
    a = torch.tensor([-1.0, 2.0], device="npu")
    out = a.relu()
    print("device", out.device)
    cpu = out.to("cpu")
    print("cpu", cpu.numpy())


if __name__ == "__main__":
    main()
