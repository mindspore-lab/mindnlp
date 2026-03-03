import mindtorch_v2 as torch
import acl


def main():
    a = torch.tensor([-1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")
    _ = a * b
    print("mul ok")
    acl.rt.synchronize_stream(0)
    relu = a.relu()
    print("relu ok", relu.device)


if __name__ == "__main__":
    main()
