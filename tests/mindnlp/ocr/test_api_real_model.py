"""
测试 OCR API 真实模型推理

运行前请确保：
1. 已安装所有依赖: pip install -r requirements/ocr-requirements.txt
2. 已下载 Qwen2-VL 模型（首次运行会自动下载约 4GB）
3. 设置 .env 文件: OCR_USE_MOCK_ENGINE=False

启动 API 服务：
cd src/mindnlp/ocr
python main.py

运行本测试：
python tests/mindnlp/ocr/test_api_real_model.py
"""

import sys
import requests
from pathlib import Path
from PIL import Image, ImageDraw
import io
import time


def create_test_image() -> bytes:
    """创建测试图像"""
    img = Image.new('RGB', (512, 512), color='white')
    draw = ImageDraw.Draw(img)
    
    # 绘制蓝色边框
    draw.rectangle([50, 50, 450, 450], outline='blue', width=3)
    
    # 添加文本
    draw.text((100, 200), "Hello Qwen2-VL!", fill='black')
    draw.text((100, 250), "This is a test image", fill='black')
    draw.text((100, 300), "for OCR API", fill='black')
    
    # 转换为 bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


def test_health_check(base_url: str = "http://localhost:8000"):
    """测试健康检查端点"""
    print("\n" + "="*70)
    print("测试 1: 健康检查")
    print("="*70)
    
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        assert response.status_code == 200
        print("✓ 健康检查通过")
        return True
    except Exception as e:
        print(f"✗ 健康检查失败: {e}")
        return False


def test_single_image_ocr(base_url: str = "http://localhost:8000"):
    """测试单图 OCR"""
    print("\n" + "="*70)
    print("测试 2: 单图 OCR（真实模型推理）")
    print("="*70)
    
    try:
        # 创建测试图像
        print("创建测试图像...")
        image_bytes = create_test_image()
        
        # 发送请求
        print(f"发送请求到 {base_url}/api/v1/ocr/predict")
        start_time = time.time()
        
        files = {'file': ('test.png', image_bytes, 'image/png')}
        data = {
            'output_format': 'text',
            'language': 'auto',
            'task_type': 'general',
            'confidence_threshold': 0.0
        }
        
        response = requests.post(
            f"{base_url}/api/v1/ocr/predict",
            files=files,
            data=data,
            timeout=60  # 真实模型推理需要更长时间
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n响应数据:")
            print(f"  success: {result.get('success')}")
            print(f"  model_name: {result.get('model_name')}")
            print(f"  inference_time: {result.get('inference_time', 'N/A')}s")
            print(f"  total_time: {elapsed_time:.2f}s")
            
            if result.get('texts'):
                print(f"\n识别的文本:")
                for i, text in enumerate(result['texts'], 1):
                    print(f"  [{i}] {text}")
            
            if result.get('raw_output'):
                print(f"\n原始输出:")
                print(f"  {result['raw_output'][:200]}...")
            
            if result.get('metadata'):
                print(f"\n元数据:")
                for key, value in result['metadata'].items():
                    print(f"  {key}: {value}")
            
            # 验证使用的是真实模型
            assert result.get('success') == True
            assert 'Qwen2-VL' in result.get('model_name', '')
            print("\n✓ 单图 OCR 测试通过（使用真实模型）")
            return True
        else:
            print(f"✗ 请求失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ 单图 OCR 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_ocr(base_url: str = "http://localhost:8000"):
    """测试批量 OCR"""
    print("\n" + "="*70)
    print("测试 3: 批量 OCR（真实模型推理）")
    print("="*70)
    
    try:
        # 创建多个测试图像
        print("创建 3 张测试图像...")
        images = []
        for i in range(3):
            img = Image.new('RGB', (256, 256), color=['red', 'green', 'blue'][i])
            draw = ImageDraw.Draw(img)
            draw.text((50, 100), f"Image {i+1}", fill='white')
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            images.append(('files', (f'test{i}.png', buffer.getvalue(), 'image/png')))
        
        # 发送请求
        print(f"发送批量请求到 {base_url}/api/v1/ocr/predict_batch")
        start_time = time.time()
        
        data = {
            'output_format': 'text',
            'language': 'auto',
            'task_type': 'general',
            'confidence_threshold': 0.0
        }
        
        response = requests.post(
            f"{base_url}/api/v1/ocr/predict_batch",
            files=images,
            data=data,
            timeout=120  # 批量处理需要更长时间
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n响应数据:")
            print(f"  success: {result.get('success')}")
            print(f"  total_images: {result.get('total_images')}")
            print(f"  total_time: {result.get('total_time', elapsed_time):.2f}s")
            print(f"  model_name: {result.get('model_name')}")
            
            if result.get('results'):
                print(f"\n处理结果 ({len(result['results'])} 张图像):")
                for i, res in enumerate(result['results'], 1):
                    print(f"\n  图像 {i}:")
                    print(f"    success: {res.get('success')}")
                    print(f"    inference_time: {res.get('inference_time')}s")
                    if res.get('texts'):
                        print(f"    text: {res['texts'][0][:50]}...")
            
            print("\n✓ 批量 OCR 测试通过（使用真实模型）")
            return True
        else:
            print(f"✗ 请求失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ 批量 OCR 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "="*70)
    print("OCR API 真实模型测试")
    print("="*70)
    
    base_url = "http://localhost:8000"
    
    # 检查 API 是否运行
    print(f"\n检查 API 服务 ({base_url})...")
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        if response.status_code != 200:
            print(f"\n✗ API 服务未正常运行")
            print("请先启动 API 服务:")
            print("  cd src/mindnlp/ocr")
            print("  python main.py")
            sys.exit(1)
        print("✓ API 服务运行中")
    except requests.exceptions.ConnectionError:
        print(f"\n✗ 无法连接到 API 服务 ({base_url})")
        print("请先启动 API 服务:")
        print("  cd src/mindnlp/ocr")
        print("  python main.py")
        sys.exit(1)
    
    # 运行测试
    results = []
    results.append(("健康检查", test_health_check(base_url)))
    results.append(("单图 OCR", test_single_image_ocr(base_url)))
    results.append(("批量 OCR", test_batch_ocr(base_url)))
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！真实模型已成功接入 API！")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
