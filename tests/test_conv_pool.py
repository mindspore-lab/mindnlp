"""Verification tests for conv and pooling operations."""
import sys
import os

# Ensure src is on the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.nn.functional as F

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} {detail}")


def test_conv2d():
    print("\n=== Conv2d ===")
    # Module
    conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 8, 8)
    y = conv(x)
    check("Conv2d output shape", y.shape == (2, 16, 8, 8), f"got {y.shape}")

    # Functional
    w = torch.randn(16, 3, 3, 3)
    y2 = F.conv2d(x, w, padding=1)
    check("F.conv2d output shape", y2.shape == (2, 16, 8, 8), f"got {y2.shape}")

    # stride=2
    conv2 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
    y3 = conv2(x)
    check("Conv2d stride=2", y3.shape == (2, 16, 4, 4), f"got {y3.shape}")

    # No padding
    conv3 = nn.Conv2d(3, 8, 3)
    y4 = conv3(x)
    check("Conv2d no padding", y4.shape == (2, 8, 6, 6), f"got {y4.shape}")

    # Groups
    conv4 = nn.Conv2d(4, 8, 3, padding=1, groups=2)
    x4 = torch.randn(1, 4, 6, 6)
    y5 = conv4(x4)
    check("Conv2d groups=2", y5.shape == (1, 8, 6, 6), f"got {y5.shape}")

    # With bias
    conv5 = nn.Conv2d(3, 4, 1, bias=True)
    y6 = conv5(torch.randn(1, 3, 4, 4))
    check("Conv2d 1x1 with bias", y6.shape == (1, 4, 4, 4), f"got {y6.shape}")


def test_conv1d():
    print("\n=== Conv1d ===")
    conv = nn.Conv1d(3, 16, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 20)
    y = conv(x)
    check("Conv1d output shape", y.shape == (2, 16, 20), f"got {y.shape}")

    # No padding
    conv2 = nn.Conv1d(3, 8, 5)
    y2 = conv2(x)
    check("Conv1d kernel=5 no pad", y2.shape == (2, 8, 16), f"got {y2.shape}")

    # stride=2
    conv3 = nn.Conv1d(3, 8, 3, stride=2, padding=1)
    y3 = conv3(x)
    check("Conv1d stride=2", y3.shape == (2, 8, 10), f"got {y3.shape}")


def test_conv_transpose2d():
    print("\n=== ConvTranspose2d ===")
    deconv = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
    x = torch.randn(2, 16, 4, 4)
    y = deconv(x)
    check("ConvTranspose2d output shape", y.shape == (2, 3, 8, 8), f"got {y.shape}")

    # stride=1
    deconv2 = nn.ConvTranspose2d(8, 4, 3, padding=1)
    y2 = deconv2(torch.randn(1, 8, 6, 6))
    check("ConvTranspose2d stride=1", y2.shape == (1, 4, 6, 6), f"got {y2.shape}")


def test_conv_transpose1d():
    print("\n=== ConvTranspose1d ===")
    deconv = nn.ConvTranspose1d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
    x = torch.randn(2, 8, 10)
    y = deconv(x)
    check("ConvTranspose1d output shape", y.shape == (2, 3, 20), f"got {y.shape}")


def test_max_pool2d():
    print("\n=== MaxPool2d ===")
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    x = torch.randn(2, 3, 8, 8)
    y = pool(x)
    check("MaxPool2d output shape", y.shape == (2, 3, 4, 4), f"got {y.shape}")

    # with padding
    pool2 = nn.MaxPool2d(3, stride=1, padding=1)
    y2 = pool2(x)
    check("MaxPool2d padding=1", y2.shape == (2, 3, 8, 8), f"got {y2.shape}")

    # Functional
    y3 = F.max_pool2d(x, 2, 2)
    check("F.max_pool2d", y3.shape == (2, 3, 4, 4), f"got {y3.shape}")

    # Verify max values are correct
    x_small = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # (1, 1, 2, 2)
    y4 = F.max_pool2d(x_small, 2, 2)
    check("MaxPool2d value correctness", abs(y4.tolist()[0][0][0][0] - 4.0) < 1e-5,
          f"got {y4.tolist()}")


def test_avg_pool2d():
    print("\n=== AvgPool2d ===")
    pool = nn.AvgPool2d(kernel_size=2, stride=2)
    x = torch.randn(2, 3, 8, 8)
    y = pool(x)
    check("AvgPool2d output shape", y.shape == (2, 3, 4, 4), f"got {y.shape}")

    # Functional
    y2 = F.avg_pool2d(x, 2, 2)
    check("F.avg_pool2d", y2.shape == (2, 3, 4, 4), f"got {y2.shape}")

    # Verify average values are correct
    x_small = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # (1, 1, 2, 2)
    y3 = F.avg_pool2d(x_small, 2, 2)
    check("AvgPool2d value correctness", abs(y3.tolist()[0][0][0][0] - 2.5) < 1e-5,
          f"got {y3.tolist()}")


def test_adaptive_avg_pool2d():
    print("\n=== AdaptiveAvgPool2d ===")
    pool = nn.AdaptiveAvgPool2d((1, 1))
    x = torch.randn(2, 3, 8, 8)
    y = pool(x)
    check("AdaptiveAvgPool2d (1,1) shape", y.shape == (2, 3, 1, 1), f"got {y.shape}")

    pool2 = nn.AdaptiveAvgPool2d((4, 4))
    y2 = pool2(x)
    check("AdaptiveAvgPool2d (4,4) shape", y2.shape == (2, 3, 4, 4), f"got {y2.shape}")

    # Functional
    y3 = F.adaptive_avg_pool2d(x, (2, 2))
    check("F.adaptive_avg_pool2d", y3.shape == (2, 3, 2, 2), f"got {y3.shape}")

    # Integer output_size
    y4 = F.adaptive_avg_pool2d(x, 1)
    check("F.adaptive_avg_pool2d int arg", y4.shape == (2, 3, 1, 1), f"got {y4.shape}")


if __name__ == "__main__":
    test_conv2d()
    test_conv1d()
    test_conv_transpose2d()
    test_conv_transpose1d()
    test_max_pool2d()
    test_avg_pool2d()
    test_adaptive_avg_pool2d()

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
