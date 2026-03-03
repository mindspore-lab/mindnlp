import mindtorch_v2 as torch


def main():
    a = torch.tensor([-1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")
    prod = a * b
    print("mul ok", prod.to("cpu").numpy())
    relu = a.relu()
    print("relu ok", relu.to("cpu").numpy())


if __name__ == "__main__":
    main()
