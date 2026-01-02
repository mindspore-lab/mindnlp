"""
预处理组件单元测试
测试 ImageProcessor, PromptBuilder, BatchCollator, InputValidator
"""

import pytest
import numpy as np
import torch
from PIL import Image
import io
from pathlib import Path
import tempfile
import yaml
import sys
import os
from pathlib import Path

# Mock mindspore 和 mindtorch 以避免依赖问题
from unittest.mock import MagicMock

# 在导入 mindnlp 之前 mock 掉 mindspore 和 mindtorch 所有相关模块
mock_mindspore = MagicMock()
mock_mindspore.communication = MagicMock()
mock_mindspore.communication.GlobalComm = MagicMock()

mock_mindtorch = MagicMock()
mock_mindtorch.configs = MagicMock()
mock_mindtorch.configs.SUPPORT_BF16 = False
mock_mindtorch.configs.SOC = 'mock'

sys.modules['mindspore'] = mock_mindspore
sys.modules['mindspore.communication'] = mock_mindspore.communication
sys.modules['mindtorch'] = mock_mindtorch
sys.modules['mindtorch.serialization'] = MagicMock()
sys.modules['mindtorch.nn'] = MagicMock()
sys.modules['mindtorch.nn.parallel'] = MagicMock()
sys.modules['mindtorch.nn.parallel.distributed'] = MagicMock()
sys.modules['mindtorch.configs'] = mock_mindtorch.configs

# 使用绝对导入
from mindnlp.ocr.core.processor.image import ImageProcessor
from mindnlp.ocr.core.processor.prompt import PromptBuilder
from mindnlp.ocr.core.processor.batch import BatchCollator
from mindnlp.ocr.core.validator.input import InputValidator


class TestImageProcessor:
    """测试 ImageProcessor"""
    
    @pytest.fixture
    def processor(self):
        """创建处理器"""
        return ImageProcessor(target_size=(448, 448))
    
    @pytest.fixture
    def rgb_image(self):
        """创建 RGB 测试图像"""
        return Image.new('RGB', (800, 600), color=(255, 0, 0))
    
    @pytest.fixture
    def rgba_image(self):
        """创建 RGBA 测试图像"""
        return Image.new('RGBA', (800, 600), color=(255, 0, 0, 200))
    
    @pytest.fixture
    def grayscale_image(self):
        """创建灰度测试图像"""
        return Image.new('L', (800, 600), color=128)
    
    def test_process_pil_image(self, processor, rgb_image):
        """测试处理 PIL 图像"""
        tensor, transform_info = processor.process(rgb_image)
        
        # 检查输出
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 448, 448)
        assert isinstance(transform_info, dict)
        
        # 检查 transform_info
        assert 'original_size' in transform_info
        assert 'resized_size' in transform_info
        assert 'target_size' in transform_info
        assert 'scale' in transform_info
        assert 'padding' in transform_info
        assert 'offset' in transform_info
        
        assert transform_info['original_size'] == (800, 600)
        assert transform_info['target_size'] == (448, 448)
    
    def test_process_bytes(self, processor, rgb_image):
        """测试处理字节数据"""
        # 转换为字节
        buffer = io.BytesIO()
        rgb_image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        
        tensor, transform_info = processor.process(image_bytes)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 448, 448)
    
    def test_process_numpy_array(self, processor, rgb_image):
        """测试处理 NumPy 数组"""
        # 转换为 NumPy 数组
        image_array = np.array(rgb_image)
        
        tensor, transform_info = processor.process(image_array)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 448, 448)
    
    def test_process_rgba_image(self, processor, rgba_image):
        """测试处理 RGBA 图像"""
        tensor, transform_info = processor.process(rgba_image)
        
        # 应该转换为 RGB
        assert tensor.shape == (1, 3, 448, 448)
    
    def test_process_grayscale_image(self, processor, grayscale_image):
        """测试处理灰度图像"""
        tensor, transform_info = processor.process(grayscale_image)
        
        # 应该转换为 RGB
        assert tensor.shape == (1, 3, 448, 448)
    
    def test_extreme_aspect_ratio_wide(self, processor):
        """测试极端宽高比（宽图）"""
        wide_image = Image.new('RGB', (2000, 100), color=(0, 255, 0))
        tensor, transform_info = processor.process(wide_image)
        
        assert tensor.shape == (1, 3, 448, 448)
        # 宽图应该有上下 padding
        assert transform_info['padding']['top'] > 0
        assert transform_info['padding']['bottom'] > 0
    
    def test_extreme_aspect_ratio_tall(self, processor):
        """测试极端宽高比（高图）"""
        tall_image = Image.new('RGB', (100, 2000), color=(0, 0, 255))
        tensor, transform_info = processor.process(tall_image)
        
        assert tensor.shape == (1, 3, 448, 448)
        # 高图应该有左右 padding
        assert transform_info['padding']['left'] > 0
        assert transform_info['padding']['right'] > 0
    
    def test_restore_coordinates(self, processor, rgb_image):
        """测试坐标恢复"""
        _, transform_info = processor.process(rgb_image)
        
        # 模型输出坐标 (假设检测到的框)
        model_coords = np.array([
            [100, 100, 200, 200],  # x1, y1, x2, y2
            [250, 250, 350, 350]
        ])
        
        original_coords = processor.restore_coordinates(model_coords, transform_info)
        
        assert isinstance(original_coords, np.ndarray)
        assert original_coords.shape == model_coords.shape
        # 坐标应该被放大到原图尺寸
        assert np.all(original_coords >= 0)
    
    def test_normalization(self, processor, rgb_image):
        """测试归一化值范围"""
        tensor, _ = processor.process(rgb_image)
        
        # 检查值范围 (归一化后应该大致在 [-2, 2] 之间)
        assert tensor.min() >= -3.0
        assert tensor.max() <= 3.0
    
    def test_invalid_input(self, processor):
        """测试无效输入"""
        from mindnlp.ocr.core.exceptions import ValidationError
        with pytest.raises(ValidationError):
            processor.process("invalid_image_data")
    
    def test_empty_image(self, processor):
        """测试空图像"""
        from mindnlp.ocr.core.exceptions import ValidationError
        empty_image = Image.new('RGB', (0, 0))
        with pytest.raises(ValidationError):
            processor.process(empty_image)


class TestPromptBuilder:
    """测试 PromptBuilder"""
    
    @pytest.fixture
    def builder(self):
        """创建构建器（使用默认模板）"""
        return PromptBuilder()
    
    @pytest.fixture
    def custom_yaml_file(self, tmp_path):
        """创建自定义 YAML 模板文件"""
        yaml_content = {
            'task_prompts': {
                'general': {
                    'zh': '自定义通用OCR提示',
                    'en': 'Custom general OCR prompt'
                }
            },
            'format_prompts': {
                'text': {
                    'zh': '自定义文本格式',
                    'en': 'Custom text format'
                }
            }
        }
        
        yaml_file = tmp_path / "custom_prompts.yaml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True)
        
        return yaml_file
    
    def test_build_general_prompt_zh(self, builder):
        """测试构建通用中文提示"""
        prompt = builder.build(task_type='general', language='zh', output_format='text')
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert 'OCR' in prompt or '识别' in prompt
    
    def test_build_general_prompt_en(self, builder):
        """测试构建通用英文提示"""
        prompt = builder.build(task_type='general', language='en', output_format='text')
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert 'OCR' in prompt.upper() or 'RECOGNIZE' in prompt.upper()
    
    def test_build_document_prompt(self, builder):
        """测试构建文档提示"""
        prompt = builder.build(task_type='document', language='zh', output_format='json')
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    def test_build_table_prompt(self, builder):
        """测试构建表格提示"""
        prompt = builder.build(task_type='table', language='zh', output_format='markdown')
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    def test_build_formula_prompt(self, builder):
        """测试构建公式提示"""
        prompt = builder.build(task_type='formula', language='en', output_format='text')
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    def test_multi_language_support(self, builder):
        """测试多语言支持"""
        languages = ['zh', 'en', 'ja', 'ko']
        
        for lang in languages:
            prompt = builder.build(task_type='general', language=lang, output_format='text')
            assert isinstance(prompt, str)
            assert len(prompt) > 0
    
    def test_custom_prompt_variables(self, builder):
        """测试自定义提示变量"""
        # 使用带变量的自定义提示
        custom_prompt = "识别图像中的 {target_type}，输出格式为 {format}"
        prompt = builder.build(
            task_type='general',
            language='zh',
            output_format='text',
            custom_prompt=custom_prompt,
            target_type='文字',
            format='纯文本'
        )
        
        assert '文字' in prompt
        assert '纯文本' in prompt
    
    def test_load_custom_yaml(self, custom_yaml_file):
        """测试加载自定义 YAML 模板"""
        builder = PromptBuilder(template_file=custom_yaml_file)
        prompt = builder.build(task_type='general', language='zh', output_format='text')
        
        assert '自定义通用OCR提示' in prompt
        assert '自定义文本格式' in prompt
    
    def test_invalid_task_type(self, builder):
        """测试无效任务类型"""
        # 应该使用默认任务类型
        prompt = builder.build(task_type='invalid_task', language='zh', output_format='text')
        assert isinstance(prompt, str)
    
    def test_invalid_language(self, builder):
        """测试无效语言"""
        # 应该使用默认语言
        prompt = builder.build(task_type='general', language='invalid_lang', output_format='text')
        assert isinstance(prompt, str)
    
    def test_invalid_format(self, builder):
        """测试无效格式"""
        # 应该使用默认格式
        prompt = builder.build(task_type='general', language='zh', output_format='invalid_format')
        assert isinstance(prompt, str)


class TestBatchCollator:
    """测试 BatchCollator"""
    
    @pytest.fixture
    def collator(self):
        """创建收集器"""
        return BatchCollator()
    
    def test_collate_single_tensor(self, collator):
        """测试收集单个 Tensor"""
        tensor = torch.randn(3, 448, 448)
        batch = collator.collate([tensor])
        
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (1, 3, 448, 448)
    
    def test_collate_multiple_tensors(self, collator):
        """测试收集多个 Tensor"""
        tensors = [
            torch.randn(3, 448, 448),
            torch.randn(3, 448, 448),
            torch.randn(3, 448, 448)
        ]
        batch = collator.collate(tensors)
        
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (3, 3, 448, 448)
    
    def test_group_by_size_similar_aspect_ratio(self, collator):
        """测试按相似宽高比分组"""
        # 创建相似宽高比的图像
        sizes = [
            (800, 600),   # 1.33
            (1024, 768),  # 1.33
            (640, 480),   # 1.33
            (600, 800),   # 0.75
            (768, 1024),  # 0.75
        ]
        
        groups = collator.group_by_size(sizes, max_group_diff=0.2)
        
        assert isinstance(groups, list)
        # 应该分成两组：横向和纵向
        assert len(groups) >= 2
    
    def test_group_by_size_diverse_aspect_ratio(self, collator):
        """测试按不同宽高比分组"""
        sizes = [
            (1000, 100),  # 10.0 (极宽)
            (800, 600),   # 1.33
            (100, 1000),  # 0.1 (极高)
            (600, 600),   # 1.0 (正方形)
        ]
        
        groups = collator.group_by_size(sizes, max_group_diff=0.2)
        
        assert isinstance(groups, list)
        # 应该分成多个组
        assert len(groups) >= 3
    
    def test_smart_padding_alignment(self, collator):
        """测试智能 Padding 对齐"""
        # 测试 32 像素对齐
        size = (450, 350)
        padded_size = collator.smart_padding([size], target_size=(448, 448))
        
        assert isinstance(padded_size, tuple)
        assert len(padded_size) == 2
        # 应该对齐到 32 的倍数
        assert padded_size[0] % 32 == 0
        assert padded_size[1] % 32 == 0
        assert padded_size[0] >= 450
        assert padded_size[1] >= 350
    
    def test_smart_padding_multiple_sizes(self, collator):
        """测试多个尺寸的智能 Padding"""
        sizes = [
            (400, 300),
            (450, 350),
            (500, 400)
        ]
        padded_size = collator.smart_padding(sizes, target_size=(448, 448))
        
        # 应该能容纳所有图像
        assert padded_size[0] >= 500
        assert padded_size[1] >= 400
        # 应该对齐到 32 的倍数
        assert padded_size[0] % 32 == 0
        assert padded_size[1] % 32 == 0
    
    def test_empty_batch(self, collator):
        """测试空批次"""
        with pytest.raises((ValueError, IndexError)):
            collator.collate([])
    
    def test_mismatched_channels(self, collator):
        """测试通道不匹配"""
        tensors = [
            torch.randn(3, 448, 448),
            torch.randn(1, 448, 448),  # 不同通道数
        ]
        with pytest.raises((ValueError, RuntimeError)):
            collator.collate(tensors)


class TestInputValidator:
    """测试 InputValidator"""
    
    @pytest.fixture
    def validator(self):
        """创建验证器"""
        return InputValidator(
            max_file_size=10 * 1024 * 1024,  # 10MB
            max_image_size=(4096, 4096),
            allowed_formats=('JPEG', 'PNG', 'BMP', 'TIFF')
        )
    
    @pytest.fixture
    def valid_image_bytes(self):
        """创建有效的图像字节"""
        image = Image.new('RGB', (800, 600), color=(255, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()
    
    def test_validate_valid_image(self, validator, valid_image_bytes):
        """测试验证有效图像"""
        result = validator.validate_image(valid_image_bytes)
        assert result is True
    
    def test_validate_image_formats(self, validator):
        """测试验证不同图像格式"""
        formats = ['JPEG', 'PNG', 'BMP', 'TIFF']
        
        for fmt in formats:
            image = Image.new('RGB', (800, 600))
            buffer = io.BytesIO()
            image.save(buffer, format=fmt)
            image_bytes = buffer.getvalue()
            
            result = validator.validate_image(image_bytes)
            assert result is True
    
    def test_validate_oversized_file(self, validator):
        """测试验证超大文件"""
        # 创建超过 10MB 的图像
        large_image = Image.new('RGB', (5000, 5000))
        buffer = io.BytesIO()
        large_image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            validator.validate_image(image_bytes)
    
    def test_validate_oversized_dimensions(self, validator):
        """测试验证超大尺寸"""
        # 创建超过 4096x4096 的图像
        large_image = Image.new('RGB', (5000, 5000))
        buffer = io.BytesIO()
        # 使用压缩来避免文件大小限制
        large_image.save(buffer, format='JPEG', quality=10)
        image_bytes = buffer.getvalue()
        
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            validator.validate_image(image_bytes)
    
    def test_validate_invalid_format(self, validator):
        """测试验证无效格式"""
        # GIF 不在允许的格式中
        image = Image.new('RGB', (800, 600))
        buffer = io.BytesIO()
        image.save(buffer, format='GIF')
        image_bytes = buffer.getvalue()
        
        with pytest.raises(ValueError, match="not in allowed formats"):
            validator.validate_image(image_bytes)
    
    def test_validate_corrupted_image(self, validator):
        """测试验证损坏的图像"""
        corrupted_bytes = b'not an image'
        
        with pytest.raises(ValueError, match="Invalid image data"):
            validator.validate_image(corrupted_bytes)
    
    def test_validate_valid_params(self, validator):
        """测试验证有效参数"""
        result = validator.validate_params(
            output_format='json',
            language='zh',
            task_type='general'
        )
        assert result is True
    
    def test_validate_all_output_formats(self, validator):
        """测试验证所有输出格式"""
        formats = ['text', 'json', 'markdown']
        
        for fmt in formats:
            result = validator.validate_params(
                output_format=fmt,
                language='zh',
                task_type='general'
            )
            assert result is True
    
    def test_validate_all_languages(self, validator):
        """测试验证所有语言"""
        languages = ['auto', 'zh', 'en', 'ja', 'ko']
        
        for lang in languages:
            result = validator.validate_params(
                output_format='text',
                language=lang,
                task_type='general'
            )
            assert result is True
    
    def test_validate_all_task_types(self, validator):
        """测试验证所有任务类型"""
        tasks = ['general', 'document', 'table', 'formula']
        
        for task in tasks:
            result = validator.validate_params(
                output_format='text',
                language='zh',
                task_type=task
            )
            assert result is True
    
    def test_validate_invalid_output_format(self, validator):
        """测试验证无效输出格式"""
        with pytest.raises(ValueError, match="output_format must be one of"):
            validator.validate_params(
                output_format='invalid',
                language='zh',
                task_type='general'
            )
    
    def test_validate_invalid_language(self, validator):
        """测试验证无效语言"""
        with pytest.raises(ValueError, match="language must be one of"):
            validator.validate_params(
                output_format='text',
                language='invalid',
                task_type='general'
            )
    
    def test_validate_invalid_task_type(self, validator):
        """测试验证无效任务类型"""
        with pytest.raises(ValueError, match="task_type must be one of"):
            validator.validate_params(
                output_format='text',
                language='zh',
                task_type='invalid'
            )


class TestIntegration:
    """集成测试"""
    
    def test_full_preprocessing_pipeline(self):
        """测试完整预处理流程"""
        # 创建组件
        validator = InputValidator()
        processor = ImageProcessor(target_size=(448, 448))
        builder = PromptBuilder()
        collator = BatchCollator()
        
        # 1. 创建测试图像
        image = Image.new('RGB', (800, 600), color=(255, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        
        # 2. 验证输入
        validator.validate_image(image_bytes)
        validator.validate_params(
            output_format='json',
            language='zh',
            task_type='general'
        )
        
        # 3. 处理图像
        tensor, transform_info = processor.process(image)
        
        # 4. 构建提示
        prompt = builder.build(
            task_type='general',
            language='zh',
            output_format='json'
        )
        
        # 5. 批处理
        batch = collator.collate([tensor])
        
        # 验证结果
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (1, 3, 448, 448)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert isinstance(transform_info, dict)
    
    def test_batch_processing_multiple_images(self):
        """测试批量处理多个图像"""
        processor = ImageProcessor(target_size=(448, 448))
        collator = BatchCollator()
        
        # 创建不同尺寸的图像
        images = [
            Image.new('RGB', (800, 600)),
            Image.new('RGB', (640, 480)),
            Image.new('RGB', (1024, 768)),
        ]
        
        # 处理所有图像
        tensors = []
        transform_infos = []
        for image in images:
            tensor, transform_info = processor.process(image)
            tensors.append(tensor.squeeze(0))  # 移除批次维度
            transform_infos.append(transform_info)
        
        # 批处理
        batch = collator.collate(tensors)
        
        assert batch.shape == (3, 3, 448, 448)
    
    def test_coordinate_restoration_accuracy(self):
        """测试坐标恢复精度"""
        processor = ImageProcessor(target_size=(448, 448))
        
        # 创建已知尺寸的图像
        original_size = (1000, 800)
        image = Image.new('RGB', original_size)
        
        # 处理图像
        _, transform_info = processor.process(image)
        
        # 模拟模型输出的坐标 (在 448x448 图像上)
        model_coords = np.array([[224, 224, 336, 336]])  # 中心区域
        
        # 恢复到原图坐标
        original_coords = processor.restore_coordinates(model_coords, transform_info)
        
        # 验证坐标在合理范围内
        assert np.all(original_coords >= 0)
        assert np.all(original_coords[:, [0, 2]] <= original_size[0])
        assert np.all(original_coords[:, [1, 3]] <= original_size[1])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
