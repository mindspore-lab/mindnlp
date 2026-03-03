import faulthandler

import mindtorch_v2 as torch


def main():
    faulthandler.enable()
    a = torch.tensor([-1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")
    print("before mul", flush=True)
    _ = a * b
    print("mul ok", flush=True)
    print("before relu", flush=True)
    relu = a.relu()
    print("relu ok", relu.device, flush=True)


if __name__ == "__main__":
    main()
