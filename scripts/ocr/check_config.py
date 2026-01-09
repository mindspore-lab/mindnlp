#!/usr/bin/env python3
"""检查OCR配置是否正确"""

import os
import sys

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from mindnlp.ocr.config.settings import get_settings

print("=" * 60)
print("OCR配置检查")
print("=" * 60)

# 获取设置
settings = get_settings()

print(f"\n📋 当前配置:")
print(f"  - 模型: {settings.default_model}")
print(f"  - 设备: {settings.device}")
print(f"  - LoRA路径: {settings.lora_weights_path}")
print(f"  - API Host: {settings.api_host}")
print(f"  - API Port: {settings.api_port}")
print(f"  - 日志级别: {settings.log_level}")

print(f"\n🔍 环境变量:")
print(f"  - OCR_LORA_WEIGHTS_PATH: {os.getenv('OCR_LORA_WEIGHTS_PATH', '未设置')}")
print(f"  - OCR_DEFAULT_MODEL: {os.getenv('OCR_DEFAULT_MODEL', '未设置')}")
print(f"  - OCR_DEVICE: {os.getenv('OCR_DEVICE', '未设置')}")

print("\n" + "=" * 60)

if settings.lora_weights_path:
    print(f"✅ LoRA路径已配置: {settings.lora_weights_path}")
    # 检查路径是否存在
    import pathlib
    lora_path = pathlib.Path(settings.lora_weights_path)
    if lora_path.exists():
        print(f"✅ LoRA路径存在")
        adapter_file = lora_path / "adapter_model.npz"
        if adapter_file.exists():
            print(f"✅ adapter_model.npz 文件存在")
        else:
            print(f"❌ adapter_model.npz 文件不存在")
    else:
        print(f"❌ LoRA路径不存在")
else:
    print("⚠️  LoRA路径未配置（将使用基础模型）")

print("=" * 60)
