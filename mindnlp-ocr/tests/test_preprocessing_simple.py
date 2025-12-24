"""
预处理组件单元测试（简化版）
仅测试不需要 torch 的组件
"""

import pytest
from PIL import Image
import io
from pathlib import Path
import sys
import tempfile
import yaml

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 直接导入模块文件，避免 __init__.py 触发 torch 导入
import importlib.util

# 导入 PromptBuilder
spec = importlib.util.spec_from_file_location(
    "prompt_builder", 
    Path(__file__).parent.parent / "core" / "processor" / "prompt.py"
)
prompt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt_module)
PromptBuilder = prompt_module.PromptBuilder

# 导入 InputValidator
spec = importlib.util.spec_from_file_location(
    "input_validator",
    Path(__file__).parent.parent / "core" / "validator" / "input.py"
)
validator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator_module)
InputValidator = validator_module.InputValidator


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
        print(f"中文提示: {prompt}")
    
    def test_build_general_prompt_en(self, builder):
        """测试构建通用英文提示"""
        prompt = builder.build(task_type='general', language='en', output_format='text')
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        print(f"英文提示: {prompt}")
    
    def test_build_document_prompt(self, builder):
        """测试构建文档提示"""
        prompt = builder.build(task_type='document', language='zh', output_format='json')
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        print(f"文档提示: {prompt}")
    
    def test_build_table_prompt(self, builder):
        """测试构建表格提示"""
        prompt = builder.build(task_type='table', language='zh', output_format='markdown')
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        print(f"表格提示: {prompt}")
    
    def test_build_formula_prompt(self, builder):
        """测试构建公式提示"""
        prompt = builder.build(task_type='formula', language='en', output_format='text')
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        print(f"公式提示: {prompt}")
    
    def test_multi_language_support(self, builder):
        """测试多语言支持"""
        languages = ['zh', 'en', 'ja', 'ko']
        
        for lang in languages:
            prompt = builder.build(task_type='general', language=lang, output_format='text')
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            print(f"{lang} 提示: {prompt}")
    
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
        print(f"自定义提示: {prompt}")
    
    def test_load_custom_yaml(self, custom_yaml_file):
        """测试加载自定义 YAML 模板"""
        builder = PromptBuilder(template_file=custom_yaml_file)
        prompt = builder.build(task_type='general', language='zh', output_format='text')
        
        assert '自定义通用OCR提示' in prompt
        assert '自定义文本格式' in prompt
        print(f"YAML 加载提示: {prompt}")
    
    def test_invalid_task_type(self, builder):
        """测试无效任务类型"""
        # 应该使用默认任务类型
        prompt = builder.build(task_type='invalid_task', language='zh', output_format='text')
        assert isinstance(prompt, str)
        print(f"无效任务类型提示: {prompt}")
    
    def test_invalid_language(self, builder):
        """测试无效语言"""
        # 应该使用默认语言
        prompt = builder.build(task_type='general', language='invalid_lang', output_format='text')
        assert isinstance(prompt, str)
        print(f"无效语言提示: {prompt}")
    
    def test_invalid_format(self, builder):
        """测试无效格式"""
        # 应该使用默认格式
        prompt = builder.build(task_type='general', language='zh', output_format='invalid_format')
        assert isinstance(prompt, str)
        print(f"无效格式提示: {prompt}")


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
        print("✓ 有效图像验证通过")
    
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
            print(f"✓ {fmt} 格式验证通过")
    
    def test_validate_oversized_file(self, validator):
        """测试验证超大文件"""
        # 创建超过 10MB 的图像
        large_image = Image.new('RGB', (5000, 5000))
        buffer = io.BytesIO()
        large_image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            validator.validate_image(image_bytes)
        print("✓ 超大文件被正确拒绝")
    
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
        print("✓ 超大尺寸被正确拒绝")
    
    def test_validate_invalid_format(self, validator):
        """测试验证无效格式"""
        # GIF 不在允许的格式中
        image = Image.new('RGB', (800, 600))
        buffer = io.BytesIO()
        image.save(buffer, format='GIF')
        image_bytes = buffer.getvalue()
        
        with pytest.raises(ValueError, match="not in allowed formats"):
            validator.validate_image(image_bytes)
        print("✓ 无效格式被正确拒绝")
    
    def test_validate_corrupted_image(self, validator):
        """测试验证损坏的图像"""
        corrupted_bytes = b'not an image'
        
        with pytest.raises(ValueError, match="Invalid image data"):
            validator.validate_image(corrupted_bytes)
        print("✓ 损坏图像被正确拒绝")
    
    def test_validate_valid_params(self, validator):
        """测试验证有效参数"""
        result = validator.validate_params(
            output_format='json',
            language='zh',
            task_type='general'
        )
        assert result is True
        print("✓ 有效参数验证通过")
    
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
            print(f"✓ {fmt} 格式验证通过")
    
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
            print(f"✓ {lang} 语言验证通过")
    
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
            print(f"✓ {task} 任务类型验证通过")
    
    def test_validate_invalid_output_format(self, validator):
        """测试验证无效输出格式"""
        with pytest.raises(ValueError, match="output_format must be one of"):
            validator.validate_params(
                output_format='invalid',
                language='zh',
                task_type='general'
            )
        print("✓ 无效输出格式被正确拒绝")
    
    def test_validate_invalid_language(self, validator):
        """测试验证无效语言"""
        with pytest.raises(ValueError, match="language must be one of"):
            validator.validate_params(
                output_format='text',
                language='invalid',
                task_type='general'
            )
        print("✓ 无效语言被正确拒绝")
    
    def test_validate_invalid_task_type(self, validator):
        """测试验证无效任务类型"""
        with pytest.raises(ValueError, match="task_type must be one of"):
            validator.validate_params(
                output_format='text',
                language='zh',
                task_type='invalid'
            )
        print("✓ 无效任务类型被正确拒绝")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
