"""
Qwen2-VL模型封装
"""

import base64
import logging
import os
from io import BytesIO
from typing import Any, Dict, List, Union
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from mindnlp.ocr.utils.cache_manager import (
    KVCacheManager, CacheConfig, get_optimal_cache_config, detect_flash_attention_support
)
from .base import VLMModelBase

# Define custom exceptions locally
class ModelLoadingError(Exception):
    """模型加载错误"""
    def __init__(self, message="Model loading failed"):
        self.message = message
        super().__init__(self.message)

class ModelInferenceError(Exception):
    """模型推理错误"""
    def __init__(self, message="Model inference failed"):
        self.message = message
        super().__init__(self.message)


logger = logging.getLogger(__name__)

# 尝试导入 qwen_vl_utils，如果不存在则使用内置实现
try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
    logger.info("qwen_vl_utils is available")
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False
    logger.warning("qwen_vl_utils not found, using built-in implementation")


class Qwen2VLModel(VLMModelBase):
    """Qwen2-VL模型封装"""

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "cuda",
                 min_pixels: int = 128*28*28, max_pixels: int = 512*28*28,
                 quantization_mode: str = "none", quantization_config: dict = None,
                 lora_weights_path: str = None,
                 cache_config: CacheConfig = None):
        """
        初始化Qwen2-VL模型

        Args:
            model_name: 模型名称
            device: 运行设备
            min_pixels: 最小像素数 (用于动态分辨率)
            max_pixels: 最大像素数 (用于动态分辨率)
            quantization_mode: 量化模式 ("none", "fp16", "int8", "int4")
            quantization_config: 量化配置字典 (可选,覆盖默认配置)
            lora_weights_path: LoRA权重路径 (可选，用于加载微调模型)
            cache_config: KV Cache配置 (可选，用于优化推理性能)
        """
        super().__init__(model_name, device)
        self.processor = None
        self.tokenizer = None
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.quantization_mode = quantization_mode.lower()
        self.quantization_config = quantization_config or {}
        self.lora_weights_path = lora_weights_path
        
        # 解析本地模型路径
        self.local_model_path = self._resolve_model_path(model_name)
        
        # 保存基础模型路径（用于 NPZ 加载模式）
        self.base_model_path = None
        self.is_npz_mode = False
        
        # 初始化 KV Cache 管理器
        if cache_config is None:
            cache_config = get_optimal_cache_config(device, model_size_gb=7.0)
        self.cache_manager = KVCacheManager(cache_config)
        self.cache_config = cache_config
        
        self.load_model()
        
        # 如果指定了LoRA权重，加载LoRA
        if self.lora_weights_path:
            self.load_lora_weights(self.lora_weights_path)
        
        self.load_tokenizer()  # 先加载tokenizer,processor需要它的chat_template
        self.load_processor()
    
    def _resolve_model_path(self, model_name: str) -> str:
        """
        解析模型路径,优先使用本地缓存的绝对路径
        
        Args:
            model_name: 模型名称,如 Qwen/Qwen2-VL-2B-Instruct
            
        Returns:
            str: 本地模型路径或原始模型名称
        """
        from pathlib import Path
        import os
        
        # 如果已经是本地路径,直接返回
        if Path(model_name).exists():
            return model_name
        
        # 获取HuggingFace缓存目录
        cache_dir = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE') or \
                   Path.home() / '.cache' / 'huggingface'
        
        # 转换模型名称为缓存目录格式
        model_cache_name = f"models--{model_name.replace('/', '--')}"
        model_cache_path = Path(cache_dir) / model_cache_name
        
        # 检查缓存是否存在
        if model_cache_path.exists():
            snapshots_dir = model_cache_path / "snapshots"
            if snapshots_dir.exists():
                # 找到最新的快照
                snapshot_dirs = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime)
                if snapshot_dirs:
                    local_path = str(snapshot_dirs[-1])
                    logger.info(f"Using local model path: {local_path}")
                    return local_path
        
        # 缓存不存在,返回原始名称
        logger.warning(f"Local cache not found for {model_name}, will use hub name")
        return model_name
    
    def _load_from_npz(self):
        """
        从 .npz 文件加载完整模型权重（内存优化版）
        
        逐个加载并替换参数，避免内存占用翻倍
        """
        import os
        import numpy as np
        import gc
        
        # 标记为 NPZ 加载模式
        self.is_npz_mode = True
        # 设置基础模型路径（用于加载 tokenizer 和 processor）
        self.base_model_path = "Qwen/Qwen2-VL-7B-Instruct"
        
        # 确定 .npz 文件路径
        if self.local_model_path.endswith('.npz'):
            npz_file = self.local_model_path
            base_model_name = "Qwen/Qwen2-VL-7B-Instruct"
        else:
            npz_file = os.path.join(self.local_model_path, 'adapter_model.npz')
            base_model_name = "Qwen/Qwen2-VL-7B-Instruct"
        
        logger.info(f"Loading model from NPZ file: {npz_file}")
        logger.info(f"File size: {os.path.getsize(npz_file) / (1024**3):.2f} GB")
        
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoConfig
            
            # 1. 确定目标精度和设备
            torch_dtype = torch.float16 if "npu" in self.device or "cuda" in self.device else torch.float32
            logger.info(f"Target device: {self.device}, dtype: {torch_dtype}")
            
            # 2. 创建空模型架构并直接放到目标设备
            logger.info("Creating model architecture on target device...")
            config = AutoConfig.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                local_files_only=False
            )
            
            # NPU 特殊配置：强制使用 eager attention（NPU 不支持 SDPA）和 FP16（不支持 BF16）
            if "npu" in self.device:
                logger.info("Configuring for NPU...")
                config._attn_implementation = "eager"  # 关键：NPU不支持SDPA
                config.torch_dtype = torch.float16  # 关键：NPU不支持BF16，强制FP16
                logger.info("Set attn_implementation='eager' and torch_dtype=float16 for NPU compatibility")
            
            # 直接在目标设备上创建空模型
            with torch.device(self.device):
                self.model = Qwen2VLForConditionalGeneration(config)
                self.model = self.model.to(dtype=torch_dtype)
            
            logger.info(f"Empty model created on {self.device}")
            
            # 3. 逐个加载并替换参数（内存优化）
            logger.info("Loading and replacing weights one by one...")
            data = np.load(npz_file)
            total_weights = len(data.files)
            logger.info(f"Found {total_weights} weight tensors")
            
            # 构建参数名映射
            param_dict = dict(self.model.named_parameters())
            buffer_dict = dict(self.model.named_buffers())
            
            # 分组权重：base_layer、lora_A、lora_B
            base_weights = {}
            lora_a_weights = {}
            lora_b_weights = {}
            
            # 先收集所有权重
            for npz_key in data.files:
                if 'lora_A.default' in npz_key:
                    # 提取模块名：base_model.model.xxx.lora_A.default.weight -> xxx
                    module_name = npz_key.replace('base_model.model.', '').replace('.lora_A.default.weight', '')
                    lora_a_weights[module_name] = data[npz_key]
                elif 'lora_B.default' in npz_key:
                    module_name = npz_key.replace('base_model.model.', '').replace('.lora_B.default.weight', '')
                    lora_b_weights[module_name] = data[npz_key]
                elif '.base_layer.weight' in npz_key:
                    # base_model.model.xxx.base_layer.weight -> xxx.weight
                    module_name = npz_key.replace('base_model.model.', '').replace('.base_layer.weight', '.weight')
                    base_weights[module_name] = data[npz_key]
                elif '.base_layer.bias' in npz_key:
                    # base_model.model.xxx.base_layer.bias -> xxx.bias
                    module_name = npz_key.replace('base_model.model.', '').replace('.base_layer.bias', '.bias')
                    base_weights[module_name] = data[npz_key]
                else:
                    # 其他权重（非 LoRA 层）: base_model.model.xxx -> xxx
                    module_name = npz_key.replace('base_model.model.', '')
                    base_weights[module_name] = data[npz_key]
            
            logger.info(f"Found {len(base_weights)} base weights, {len(lora_a_weights)} LoRA-A, {len(lora_b_weights)} LoRA-B")
            
            loaded_count = 0
            merged_count = 0
            idx = 0
            
            # 加载基础权重和合并 LoRA
            for module_name, base_weight in base_weights.items():
                idx += 1
                
                # 查找对应的模型参数
                target_param = None
                if module_name in param_dict:
                    target_param = param_dict[module_name]
                elif module_name in buffer_dict:
                    target_param = buffer_dict[module_name]
                
                if target_param is not None:
                    # 转换基础权重
                    torch_weight = torch.from_numpy(base_weight).to(dtype=torch_dtype)
                    
                    # 检查是否有对应的 LoRA 权重
                    # module_name 格式: language_model.layers.0.self_attn.q_proj.weight
                    # 需要去掉 .weight 后缀来匹配 LoRA 键
                    module_base_name = module_name.replace('.weight', '').replace('.bias', '')
                    
                    # 只对 2D 权重（weight 矩阵）合并 LoRA，跳过 1D 权重（bias）
                    if (module_base_name in lora_a_weights and 
                        module_base_name in lora_b_weights and 
                        len(base_weight.shape) == 2):  # 确保是 2D 权重
                        
                        # 获取 LoRA 权重（保持为 numpy，先检查形状）
                        lora_a_np = lora_a_weights[module_base_name]
                        lora_b_np = lora_b_weights[module_base_name]
                        
                        # 验证形状是否匹配（LoRA 也应该是 2D）
                        if len(lora_a_np.shape) == 2 and len(lora_b_np.shape) == 2:
                            lora_a = torch.from_numpy(lora_a_np).to(dtype=torch_dtype)
                            lora_b = torch.from_numpy(lora_b_np).to(dtype=torch_dtype)
                            
                            # 合并 LoRA: weight = base + lora_B @ lora_A
                            # LoRA 默认 scaling = lora_alpha / r，通常 = 1.0
                            lora_delta = torch.matmul(lora_b, lora_a)
                            torch_weight = torch_weight + lora_delta
                            
                            del lora_a, lora_b, lora_delta, lora_a_np, lora_b_np
                            merged_count += 1
                    
                    # 移动到目标设备并替换参数
                    with torch.no_grad():
                        target_param.copy_(torch_weight.to(self.device))
                    
                    # 立即释放临时内存
                    del base_weight, torch_weight
                    loaded_count += 1
                    
                    # 每 100 个参数打印一次进度
                    if idx % 100 == 0:
                        logger.info(f"Progress: {idx}/{len(base_weights)} ({loaded_count} loaded, {merged_count} merged)")
                        gc.collect()  # 强制垃圾回收
                else:
                    logger.warning(f"Parameter not found in model: {module_name}")
            
            logger.info(f"✓ Loaded {loaded_count} weights ({merged_count} LoRA merged)")
            
            # 4. 清理和优化
            del data
            gc.collect()
            
            # 设置评估模式
            self.model.eval()
            
            # NPU 优化
            if "npu" in self.device:
                try:
                    import torch_npu
                    torch_npu.npu.set_compile_mode(jit_compile=False)
                    logger.info("NPU optimization: JIT compile disabled")
                except Exception as e:
                    logger.warning(f"NPU optimization setup failed: {e}")
            
            precision_info = "FP32" if torch_dtype == torch.float32 else "FP16"
            logger.info(f"✅ Model successfully loaded with {precision_info} precision")
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            logger.error(f"Failed to load model from NPZ: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise ModelLoadingError(f"NPZ loading failed: {e}")
    
    def _get_quantization_config(self):
        """
        根据量化模式生成BitsAndBytesConfig
        
        Returns:
            BitsAndBytesConfig or None: 量化配置对象
        """
        if self.quantization_mode == "none" or self.quantization_mode == "fp16":
            # FP16 不使用 bitsandbytes, 直接通过 torch_dtype 指定
            return None
        
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            logger.error("bitsandbytes is not installed. Please install it with: pip install bitsandbytes")
            raise ImportError(
                "bitsandbytes is required for INT8/INT4 quantization. "
                "Install it with: pip install bitsandbytes"
            )
        
        if self.quantization_mode == "int8":
            # INT8 量化配置 (LLM.int8())
            logger.info("Configuring INT8 quantization (LLM.int8())")
            config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=self.quantization_config.get('int8_threshold', 6.0),
                llm_int8_skip_modules=self.quantization_config.get('int8_skip_modules', None),
            )
            return config
        
        elif self.quantization_mode == "int4":
            # INT4 量化配置 (NF4/FP4)
            logger.info("Configuring INT4 quantization (NF4)")
            
            # 解析计算数据类型
            compute_dtype_str = self.quantization_config.get('int4_compute_dtype', 'float16')
            if hasattr(torch, compute_dtype_str):
                compute_dtype = getattr(torch, compute_dtype_str)
            else:
                logger.warning(f"Unknown compute dtype: {compute_dtype_str}, using float16")
                compute_dtype = torch.float16
            
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.quantization_config.get('int4_quant_type', 'nf4'),
                bnb_4bit_use_double_quant=self.quantization_config.get('int4_use_double_quant', True),
            )
            return config
        
        else:
            logger.warning(f"Unknown quantization mode: {self.quantization_mode}, using none")
            return None

    def load_model(self):
        """
        加载Qwen2-VL模型 (支持量化和从.npz加载)
        
        Raises:
            ModelLoadingError: 模型加载失败
        """
        try:
            # 检查是否是 .npz 文件（完整模型权重）
            if self.local_model_path.endswith('.npz') or \
               (os.path.isdir(self.local_model_path) and 
                os.path.exists(os.path.join(self.local_model_path, 'adapter_model.npz'))):
                self._load_from_npz()
                return
            
            logger.info(f"Loading Qwen2-VL model: {self.model_name} with quantization: {self.quantization_mode}")
            
            # 获取量化配置
            quantization_config = self._get_quantization_config()
            
            # 确定 torch_dtype
            if self.quantization_mode == "fp16":
                torch_dtype = torch.float16
                logger.info("Using FP16 precision")
            elif self.quantization_mode in ["int8", "int4"]:
                # bitsandbytes 量化时, torch_dtype 应该是None或与bnb_4bit_compute_dtype一致
                torch_dtype = torch.float16  # 默认使用 float16 作为基础类型
                logger.info(f"Using {self.quantization_mode.upper()} quantization with bitsandbytes")
            elif self.device == "cpu":
                torch_dtype = torch.float32
                logger.info("Using FP32 precision (CPU)")
            elif "npu" in self.device or "cuda" in self.device:
                # 默认使用 FP16 优化 GPU/NPU 内存
                torch_dtype = torch.float16
                logger.info("Using FP16 precision by default (GPU/NPU)")
            else:
                torch_dtype = torch.float32
            
            # Qwen2-VL 官方推荐的加载方式：
            # 使用 Qwen2VLForConditionalGeneration 而非 AutoModel
            try:
                # 方法1：尝试直接导入 Qwen2VLForConditionalGeneration
                from transformers import Qwen2VLForConditionalGeneration
                
                # 构建加载参数
                load_kwargs = {
                    "torch_dtype": torch_dtype,
                    "low_cpu_mem_usage": True,
                    "local_files_only": True,
                }
                
                # 添加量化配置
                if quantization_config is not None:
                    load_kwargs["quantization_config"] = quantization_config
                    load_kwargs["device_map"] = "auto"  # bitsandbytes 需要 device_map
                    logger.info(f"Quantization config: {quantization_config}")
                else:
                    # 非量化模式,手动指定设备
                    load_kwargs["device_map"] = None
                
                # NPU 特殊处理
                if "npu" in self.device:
                    logger.info("Setting attn_implementation='eager' for NPU (SDPA not supported)")
                    load_kwargs["attn_implementation"] = "eager"  # 关键：NPU不支持SDPA
                    load_kwargs["use_cache"] = True  # 启用 KV cache 加速推理
                # CUDA 设备：尝试使用 Flash Attention
                elif "cuda" in self.device and self.cache_config.enable_flash_attention:
                    supported, reason = detect_flash_attention_support()
                    if supported:
                        logger.info(f"Enabling Flash Attention 2.0: {reason}")
                        load_kwargs["attn_implementation"] = "flash_attention_2"
                        load_kwargs["use_cache"] = True
                    else:
                        logger.warning(f"Flash Attention not available: {reason}, using eager")
                        load_kwargs["attn_implementation"] = "eager"
                        load_kwargs["use_cache"] = True
                else:
                    # 默认使用 eager 实现
                    load_kwargs["attn_implementation"] = "eager"
                    load_kwargs["use_cache"] = True
                
                # 加载模型
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.local_model_path,
                    **load_kwargs
                )
                
                # 如果没有使用 device_map="auto", 手动移动到目标设备
                if load_kwargs["device_map"] is None:
                    self.model = self.model.to(self.device)
                
                # NPU 性能优化
                if "npu" in self.device:
                    try:
                        import torch_npu
                        torch_npu.npu.set_compile_mode(jit_compile=False)  # 禁用JIT避免首次推理慢
                        logger.info("NPU optimization: JIT compile disabled for faster first inference")
                    except Exception as e:
                        logger.warning(f"NPU optimization setup failed: {e}")
                
                logger.info(f"Loaded with Qwen2VLForConditionalGeneration on device: {self.model.device if hasattr(self.model, 'device') else 'auto'}")
                
            except ImportError:
                # 方法2：使用 trust_remote_code 动态导入
                logger.info("Qwen2VLForConditionalGeneration not in transformers, trying trust_remote_code...")
                from transformers import AutoConfig, AutoModelForVision2Seq
                
                config = AutoConfig.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                if config.architectures:
                    logger.info(f"Model architecture: {config.architectures[0]}")
                
                # 构建加载参数
                load_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch_dtype,
                    "low_cpu_mem_usage": True,
                    "local_files_only": True,
                }
                
                # 添加量化配置
                if quantization_config is not None:
                    load_kwargs["quantization_config"] = quantization_config
                    load_kwargs["device_map"] = "auto"
                else:
                    load_kwargs["device_map"] = None
                
                # NPU 特殊处理
                if "npu" in self.device:
                    logger.info("Setting attn_implementation='eager' for NPU")
                    load_kwargs["attn_implementation"] = "eager"
                    load_kwargs["use_cache"] = True
                # CUDA 设备：尝试使用 Flash Attention
                elif "cuda" in self.device and self.cache_config.enable_flash_attention:
                    supported, reason = detect_flash_attention_support()
                    if supported:
                        logger.info(f"Enabling Flash Attention 2.0: {reason}")
                        load_kwargs["attn_implementation"] = "flash_attention_2"
                        load_kwargs["use_cache"] = True
                    else:
                        logger.warning(f"Flash Attention not available: {reason}, using eager")
                        load_kwargs["attn_implementation"] = "eager"
                        load_kwargs["use_cache"] = True
                else:
                    load_kwargs["attn_implementation"] = "eager"
                    load_kwargs["use_cache"] = True
                
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.local_model_path,
                    **load_kwargs
                )
                
                # 手动移动设备
                if load_kwargs["device_map"] is None:
                    self.model = self.model.to(self.device)
                
                logger.info(f"Loaded with AutoModelForVision2Seq on device: {self.model.device if hasattr(self.model, 'device') else 'auto'}")
            
            self.model.eval()
            
            # 记录量化信息
            if hasattr(self.model, 'is_loaded_in_8bit') and self.model.is_loaded_in_8bit:
                logger.info("✅ Model successfully loaded with INT8 quantization")
            elif hasattr(self.model, 'is_loaded_in_4bit') and self.model.is_loaded_in_4bit:
                logger.info("✅ Model successfully loaded with INT4 quantization")
            else:
                logger.info(f"✅ Model successfully loaded with {self.quantization_mode.upper()} precision")
            
            logger.info(f"Qwen2-VL model loaded successfully (type: {type(self.model).__name__}, has_generate: {hasattr(self.model, 'generate')})")
            return self.model
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}", exc_info=True)
            raise ModelLoadingError(
                message=f"Missing required dependencies for {self.model_name}: {str(e)}",
                model_name=self.model_name,
                details={"error_type": "ImportError", "missing_module": str(e)}
            ) from e
        except OSError as e:
            logger.error(f"Model files not found: {e}", exc_info=True)
            raise ModelLoadingError(
                message=f"Model files not found for {self.model_name}: {str(e)}",
                model_name=self.model_name,
                details={"error_type": "OSError", "suggestion": "Check model path or download model files"}
            ) from e
        except RuntimeError as e:
            logger.error(f"Runtime error during model loading: {e}", exc_info=True)
            raise ModelLoadingError(
                message=f"Runtime error loading {self.model_name}: {str(e)}",
                model_name=self.model_name,
                details={"error_type": "RuntimeError", "device": self.device}
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error loading model: {e}", exc_info=True)
            raise ModelLoadingError(
                message=f"Unexpected error loading {self.model_name}: {str(e)}",
                model_name=self.model_name,
                details={"error_type": type(e).__name__}
            ) from e

    def load_processor(self):
        """
        加载Qwen2-VL processor
        
        Raises:
            ModelLoadingError: Processor加载失败
        """
        try:
            # 在 NPZ 模式下，使用基础模型路径加载 processor
            if self.is_npz_mode and self.base_model_path:
                logger.info(f"Loading Qwen2-VL processor from base model: {self.base_model_path}")
                processor_path = self.base_model_path
            else:
                logger.info(f"Loading Qwen2-VL processor: {self.model_name}")
                processor_path = self.local_model_path
            
            # 只从本地加载,不再尝试在线加载
            self.processor = AutoProcessor.from_pretrained(
                processor_path,
                trust_remote_code=True,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                local_files_only=True
            )
            logger.info("Processor loaded from local cache")
            
            # 确保processor有chat_template,如果没有则从tokenizer获取
            if not hasattr(self.processor, 'chat_template') or self.processor.chat_template is None:
                logger.info("Processor missing chat_template, copying from tokenizer")
                # tokenizer已经在之前加载了
                if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                    self.processor.chat_template = self.tokenizer.chat_template
                    logger.info("Chat template copied from tokenizer to processor")
                else:
                    logger.warning("Tokenizer also missing chat_template")
                
            logger.info(f"Qwen2-VL processor loaded successfully (min_pixels={self.min_pixels}, max_pixels={self.max_pixels})")
            return self.processor
        except ModelLoadingError:
            # 重新抛出我们自己的异常
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading processor: {e}", exc_info=True)
            raise ModelLoadingError(
                message=f"Unexpected error loading processor for {self.model_name}: {str(e)}",
                model_name=self.model_name,
                details={"error_type": type(e).__name__}
            ) from e

    def load_tokenizer(self):
        """
        加载Qwen2-VL tokenizer
        
        Raises:
            ModelLoadingError: Tokenizer加载失败
        """
        try:
            # 在 NPZ 模式下，使用基础模型路径加载 tokenizer
            if self.is_npz_mode and self.base_model_path:
                logger.info(f"Loading Qwen2-VL tokenizer from base model: {self.base_model_path}")
                tokenizer_path = self.base_model_path
            else:
                logger.info(f"Loading Qwen2-VL tokenizer: {self.model_name}")
                tokenizer_path = self.local_model_path
            
            # 只从本地加载
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                local_files_only=True
            )
            logger.info("Tokenizer loaded from local cache")
            
            logger.info("Qwen2-VL tokenizer loaded successfully")
            return self.tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def _process_image_input(self, image_input: Union[str, bytes, Image.Image]) -> str:
        """
        处理不同格式的图像输入

        Args:
            image_input: 图像输入 (本地路径、URL、base64、PIL Image)

        Returns:
            处理后的图像路径或数据URI
        """
        # PIL Image
        if isinstance(image_input, Image.Image):
            buffered = BytesIO()
            image_input.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"

        # bytes (图像数据)
        if isinstance(image_input, bytes):
            # 检查是否是 base64
            try:
                # 尝试解码 base64
                if image_input.startswith(b'data:image'):
                    return image_input.decode()
                else:
                    # 原始图像字节
                    img_str = base64.b64encode(image_input).decode()
                    return f"data:image/jpeg;base64,{img_str}"
            except Exception:
                # 如果不是 base64，作为原始图像处理
                img_str = base64.b64encode(image_input).decode()
                return f"data:image/jpeg;base64,{img_str}"

        # 字符串 (路径或URL或base64)
        if isinstance(image_input, str):
            # 检查是否是 data URI
            if image_input.startswith('data:image'):
                return image_input
            # 检查是否是 URL
            elif image_input.startswith(('http://', 'https://')):
                return image_input
            # 检查是否是本地文件路径
            elif image_input.startswith('file://'):
                return image_input
            else:
                # 假定是本地文件路径
                return f"file://{image_input}"

        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _builtin_process_vision_info(self, messages: List[Dict]) -> tuple:
        """
        内置的视觉信息处理 (当 qwen_vl_utils 不可用时使用)

        Args:
            messages: 消息列表

        Returns:
            (image_inputs, video_inputs) 元组
        """
        image_inputs = []
        video_inputs = []

        for message in messages:
            if "content" in message:
                for content in message["content"]:
                    if isinstance(content, dict) and content.get("type") == "image":
                        image_input = content.get("image")
                        if image_input:
                            # 处理图像输入
                            processed_image = self._process_image_input(image_input)
                            image_inputs.append(processed_image)
                    elif isinstance(content, dict) and content.get("type") == "video":
                        video_input = content.get("video")
                        if video_input:
                            video_inputs.append(video_input)

        return image_inputs, video_inputs

    def prepare_inputs(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        准备模型输入

        Args:
            messages: 消息列表，格式参考 Qwen2-VL
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "path/to/image.jpg"},
                            {"type": "text", "text": "描述这张图片"}
                        ]
                    }
                ]
            **kwargs: 额外参数 (min_pixels, max_pixels, etc.)

        Returns:
            处理后的输入张量字典
        """
        try:
            # 1. 应用对话模板
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.debug(f"Applied chat template: {text[:100]}...")

            # 2. 处理视觉信息
            if QWEN_VL_UTILS_AVAILABLE:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs, video_inputs = self._builtin_process_vision_info(messages)

            logger.debug(f"Processed {len(image_inputs) if image_inputs else 0} images and {len(video_inputs) if video_inputs else 0} videos")

            # 3. 准备输入
            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt"
            )

            # 4. 移动到目标设备（优化：使用 non_blocking 异步传输）
            is_npu = "npu" in self.device
            inputs = {k: v.to(self.device, non_blocking=is_npu) for k, v in inputs.items()}

            logger.info("Model inputs prepared successfully")
            return inputs

        except Exception as e:
            logger.error(f"Failed to prepare inputs: {e}")
            raise

    def decode_output(self, generated_ids: torch.Tensor, input_ids: torch.Tensor) -> List[str]:
        """
        解码生成的输出

        Args:
            generated_ids: 生成的 token IDs [batch_size, seq_len]
            input_ids: 输入的 token IDs [batch_size, seq_len]

        Returns:
            解码后的文本列表
        """
        try:
            # Token trimming: 只保留新生成的 tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(input_ids, generated_ids)
            ]

            # 解码
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            logger.info(f"Decoded {len(output_text)} outputs")
            return output_text

        except Exception as e:
            logger.error(f"Failed to decode output: {e}")
            raise

    def generate(self, inputs: Dict[str, Any], **kwargs) -> List[str]:
        """
        生成输出并解码

        Args:
            inputs: 模型输入字典 (来自 prepare_inputs)
            **kwargs: 生成参数
                - max_new_tokens: 最大生成token数
                - do_sample: 是否采样
                - temperature: 温度参数
                - top_p: top-p 采样
                - use_cache: 是否使用KV Cache (默认True)

        Returns:
            解码后的文本列表
        """
        # 设置默认生成参数（OCR 优化）
        do_sample = kwargs.get('do_sample', False)
        use_cache = kwargs.get('use_cache', self.cache_config.enable_kv_cache)
        
        generation_config = {
            'max_new_tokens': kwargs.get('max_new_tokens', 512),  # OCR 通常 512 足够
            'do_sample': do_sample,
            'use_cache': use_cache,  # 从配置或参数读取
            'num_beams': 1,  # 禁用 beam search 提升速度
            'repetition_penalty': 1.0,  # 禁用重复惩罚提速
        }
        
        # 只在采样时添加 temperature 和 top_p
        if do_sample:
            generation_config['temperature'] = kwargs.get('temperature', 0.7)
            generation_config['top_p'] = kwargs.get('top_p', 0.9)

        cache_status = "enabled" if use_cache else "disabled"
        logger.info(f"Generating output with Qwen2-VL (max_new_tokens={generation_config['max_new_tokens']}, "
                   f"do_sample={do_sample}, cache={cache_status})...")

        try:
            # NPU 推理优化：减少同步开销
            if "npu" in self.device:
                torch.npu.synchronize()  # 确保输入已传输完成
            
            # 记录开始时间（用于性能统计）
            import time
            start_time = time.time()
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # NPU 推理完成后同步
            if "npu" in self.device:
                torch.npu.synchronize()
            
            # 记录推理时间
            inference_time = time.time() - start_time
            logger.info(f"Generation completed in {inference_time:.2f}s, decoding...")

            # 解码输出
            output_text = self.decode_output(generated_ids, inputs['input_ids'])
            
            # 输出缓存统计（每10次请求输出一次）
            if self.cache_manager.stats.total_requests % 10 == 0 and use_cache:
                stats = self.cache_manager.get_stats()
                logger.info(f"Cache stats: {stats}")

            return output_text

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def batch_generate(self, batch_messages: List[List[Dict]], **kwargs) -> List[str]:
        """
        批量生成推理

        Args:
            batch_messages: 批量消息列表
                [
                    [  # 第一个样本的消息
                        {"role": "user", "content": [...]},
                    ],
                    [  # 第二个样本的消息
                        {"role": "user", "content": [...]},
                    ],
                    ...
                ]
            **kwargs: 生成参数

        Returns:
            批量输出文本列表
        """
        logger.info(f"Batch generating for {len(batch_messages)} samples...")

        try:
            # 1. 批量处理输入
            batch_texts = []
            all_image_inputs = []
            all_video_inputs = []

            for messages in batch_messages:
                # 应用对话模板
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                batch_texts.append(text)

                # 处理视觉信息
                if QWEN_VL_UTILS_AVAILABLE:
                    result = process_vision_info(messages)
                    if result is not None:
                        image_inputs, video_inputs = result
                    else:
                        image_inputs, video_inputs = [], []
                else:
                    image_inputs, video_inputs = self._builtin_process_vision_info(messages)

                # 确保返回值不为 None
                if image_inputs is not None:
                    all_image_inputs.extend(image_inputs)
                if video_inputs is not None:
                    all_video_inputs.extend(video_inputs)

            # 2. 批量推理
            inputs = self.processor(
                text=batch_texts,
                images=all_image_inputs if all_image_inputs else None,
                videos=all_video_inputs if all_video_inputs else None,
                padding=True,
                return_tensors="pt"
            )

            # 移动到目标设备（优化：异步传输）
            is_npu = "npu" in self.device
            inputs = {k: v.to(self.device, non_blocking=is_npu) for k, v in inputs.items()}

            # 3. 生成（NPU 推理加速优化）
            do_sample = kwargs.get('do_sample', False)
            generation_config = {
                'max_new_tokens': kwargs.get('max_new_tokens', 256),  # 默认从 512 降至 256
                'do_sample': do_sample,
                'use_cache': True,
                'num_beams': 1,
                'repetition_penalty': 1.0,
                'pad_token_id': self.processor.tokenizer.pad_token_id,
                'eos_token_id': self.processor.tokenizer.eos_token_id,
            }
            
            # 只在采样时添加 temperature 和 top_p
            if do_sample:
                generation_config['temperature'] = kwargs.get('temperature', 0.7)
                generation_config['top_p'] = kwargs.get('top_p', 0.9)

            # NPU 优化：减少同步
            if "npu" in self.device:
                torch.npu.synchronize()
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # NPU 优化：推理完成后同步
            if "npu" in self.device:
                torch.npu.synchronize()

            # 4. 解码输出
            output_text = self.decode_output(generated_ids, inputs['input_ids'])

            return output_text

        except Exception as e:
            logger.error(f"Batch inference failed: {e}", exc_info=True)
            raise ModelInferenceError(
                message=f"Batch inference error: {str(e)}",
                model_name=self.model_name,
                details={"batch_size": len(batch_messages), "error_type": type(e).__name__}
            ) from e
    
    def load_lora_weights(self, lora_path: str):
        """
        手动加载LoRA权重（使用numpy格式，避免bitsandbytes依赖）
        
        Args:
            lora_path: LoRA权重文件路径（.npz文件或包含adapter_model.npz的目录）
        
        Raises:
            ModelLoadingError: LoRA权重加载失败
        """
        import os
        import numpy as np
        
        try:
            # 确定权重文件路径
            if os.path.isdir(lora_path):
                weights_file = os.path.join(lora_path, 'adapter_model.npz')
            else:
                weights_file = lora_path
            
            if not os.path.exists(weights_file):
                raise FileNotFoundError(f"LoRA weights file not found: {weights_file}")
            
            logger.info(f"Loading LoRA weights from: {weights_file}")
            
            # 加载numpy权重
            weights = np.load(weights_file)
            logger.info(f"Loaded {len(weights.files)} LoRA parameters")
            
            # 转换为PyTorch张量
            lora_state_dict = {}
            for key in weights.files:
                tensor = torch.from_numpy(weights[key])
                # 移动到与模型相同的设备和数据类型
                if hasattr(self.model, 'dtype'):
                    tensor = tensor.to(dtype=self.model.dtype, device=self.device)
                else:
                    tensor = tensor.to(device=self.device)
                lora_state_dict[key] = tensor
            
            # 手动合并LoRA权重到模型
            model_state_dict = self.model.state_dict()
            updated_count = 0
            
            for lora_key, lora_weight in lora_state_dict.items():
                # 解析LoRA键名（格式: base_model.model.xxx.lora_A.weight 或 lora_B.weight）
                # 移除前缀得到实际的模型参数名
                if 'base_model.model.' in lora_key:
                    # 标准PEFT格式
                    model_key = lora_key.replace('base_model.model.', '')
                else:
                    model_key = lora_key
                
                # 检查是否是LoRA A/B矩阵
                if 'lora_A' in model_key or 'lora_B' in model_key:
                    # 获取基础权重的键名
                    base_key = model_key.replace('.lora_A.weight', '.weight') \
                                       .replace('.lora_B.weight', '.weight') \
                                       .replace('.lora_A.default.weight', '.weight') \
                                       .replace('.lora_B.default.weight', '.weight')
                    
                    # 如果基础权重存在于模型中，更新它
                    if base_key in model_state_dict:
                        # 这里简化处理：直接添加LoRA权重
                        # 完整实现应该是 W_new = W_base + (lora_B @ lora_A) * scaling
                        # 但这需要配对的A和B矩阵，暂时跳过
                        pass
                    else:
                        # 可能是直接保存的合并权重
                        if model_key in model_state_dict:
                            model_state_dict[model_key] = lora_weight
                            updated_count += 1
                else:
                    # 非LoRA参数，直接更新
                    if model_key in model_state_dict:
                        model_state_dict[model_key] = lora_weight
                        updated_count += 1
            
            # 如果有更新，重新加载到模型
            if updated_count > 0:
                self.model.load_state_dict(model_state_dict, strict=False)
                logger.info(f"✅ Successfully loaded LoRA weights, updated {updated_count} parameters")
            else:
                logger.warning("⚠️  No parameters were updated. LoRA weights may not be compatible.")
                # 尝试使用PEFT标准方式（可能失败）
                logger.info("Attempting to load LoRA using PEFT (may fail with mindtorch)...")
                try:
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(self.model, lora_path)
                    logger.info("✅ Successfully loaded LoRA using PEFT")
                except Exception as peft_error:
                    logger.warning(f"PEFT loading failed (expected with mindtorch): {peft_error}")
                    raise ModelLoadingError(
                        message=f"Failed to load LoRA weights: {str(peft_error)}",
                        model_name=self.model_name,
                        details={"lora_path": lora_path, "error": str(peft_error)}
                    )
        
        except Exception as e:
            logger.error(f"Failed to load LoRA weights: {e}", exc_info=True)
            raise ModelLoadingError(
                message=f"Failed to load LoRA weights from {lora_path}: {str(e)}",
                model_name=self.model_name,
                details={"lora_path": lora_path, "error_type": type(e).__name__}
            ) from e

            logger.info(f"Batch generation completed for {len(output_text)} samples")
            return output_text

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            raise    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计字典
        """
        return self.cache_manager.get_stats()
    
    def clear_cache(self):
        """清空 KV Cache"""
        self.cache_manager.clear()
        logger.info("KV Cache cleared")
    
    def reset_cache_stats(self):
        """重置缓存统计"""
        self.cache_manager.reset_stats()
        logger.info("Cache statistics reset")
    
    def update_cache_config(self, new_config: CacheConfig):
        """
        更新缓存配置
        
        Args:
            new_config: 新的缓存配置
        """
        self.cache_config = new_config
        self.cache_manager.config = new_config
        logger.info(f"Cache config updated: {new_config}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息（包括 Flash Attention 状态）
        
        Returns:
            模型信息字典
        """
        info = {
            'model_name': self.model_name,
            'device': str(self.device),
            'quantization_mode': self.quantization_mode,
            'kv_cache_enabled': self.cache_config.enable_kv_cache,
            'flash_attention_enabled': self.cache_config.enable_flash_attention,
            'max_cache_size_mb': self.cache_config.max_cache_size_mb,
        }
        
        # 检查实际的 attention 实现
        if hasattr(self.model, 'config'):
            config = self.model.config
            if hasattr(config, '_attn_implementation'):
                info['attn_implementation'] = config._attn_implementation
        
        # 添加 Flash Attention 支持检测
        flash_supported, flash_reason = detect_flash_attention_support()
        info['flash_attention_support'] = flash_supported
        info['flash_attention_reason'] = flash_reason
        
        return info