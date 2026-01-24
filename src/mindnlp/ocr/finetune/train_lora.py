"""
Qwen2-VL LoRA Fine-tuning Script
使用 PEFT 库实现 LoRA 微调
"""
import os
import sys

# ===== 关键：在导入任何库之前检查CPU offload并隐藏NPU =====
# 这必须在mindtorch初始化之前完成
if '--cpu_offload' in sys.argv:
    # 保存原始值供后续恢复（在模型加载完成后）
    _original_npu_devices = os.environ.get('NPU_VISIBLE_DEVICES', None)
    os.environ['NPU_VISIBLE_DEVICES'] = ''  # 隐藏所有NPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 也隐藏CUDA设备
    print("⚠️  CPU offload mode: NPU hidden before module imports")
# ============================================================

import argparse
import logging
from pathlib import Path
from typing import Optional
import gc
import types
from importlib.machinery import ModuleSpec

# 在导入peft之前创建假的bitsandbytes模块（避免mindtorch兼容性问题）
if 'bitsandbytes' not in sys.modules:
    fake_bnb = types.ModuleType('bitsandbytes')
    fake_bnb.__version__ = '0.0.0'
    fake_bnb.__file__ = '/fake/path/bitsandbytes/__init__.py'
    fake_bnb.__spec__ = ModuleSpec('bitsandbytes', None, origin=fake_bnb.__file__)
    
    # 创建必要的子模块
    fake_bnb.nn = types.ModuleType('bitsandbytes.nn')
    fake_bnb.nn.__spec__ = ModuleSpec('bitsandbytes.nn', None)
    fake_bnb._ops = types.ModuleType('bitsandbytes._ops')
    fake_bnb._ops.__spec__ = ModuleSpec('bitsandbytes._ops', None)
    fake_bnb.research = types.ModuleType('bitsandbytes.research')
    fake_bnb.research.__spec__ = ModuleSpec('bitsandbytes.research', None)
    fake_bnb.utils = types.ModuleType('bitsandbytes.utils')
    fake_bnb.utils.__spec__ = ModuleSpec('bitsandbytes.utils', None)
    
    sys.modules['bitsandbytes'] = fake_bnb
    sys.modules['bitsandbytes.nn'] = fake_bnb.nn
    sys.modules['bitsandbytes._ops'] = fake_bnb._ops
    sys.modules['bitsandbytes.research'] = fake_bnb.research
    sys.modules['bitsandbytes.utils'] = fake_bnb.utils

import torch
import torch.distributed as dist

# CPU offload模式：在导入transformers前强制mindspore使用CPU
if '--cpu_offload' in sys.argv:
    try:
        import mindspore
        # 关键：设置mindspore的device_target为CPU，阻止NPU初始化
        mindspore.set_context(device_target='CPU')
        print(f"✓ MindSpore context set to CPU (device_target={mindspore.get_context('device_target')})")
        
        # 同时设置mindtorch默认设备
        from mindtorch._bind import set_default_device, get_default_device
        set_default_device('cpu')
        print(f"✓ mindtorch default device set to CPU (current: {get_default_device()})")
        
        # Monkey patch mindtorch Tensor.random_以修复CUDA设备上的未实现问题
        try:
            from mindtorch import Tensor
            from mindtorch import ops
            
            def patched_random_(self, from_=0, to=None, *, generator=None):
                """修复版本：强制在CPU执行随机操作，无论tensor原设备"""
                original_device = self.device
                import mindtorch
                
                # 始终在纯CPU tensor上执行随机操作
                cpu_tensor = mindtorch.empty_like(self, device='cpu')
                ops.inplace_random(cpu_tensor, from_, to, generator=generator)
                
                # 如果原tensor在CUDA，移回去；否则直接替换数据
                if str(original_device).startswith('cuda'):
                    self.data = cpu_tensor.to(original_device).data
                else:
                    self.data = cpu_tensor.data
                
                return self
            
            Tensor.random_ = patched_random_
            print("✓ Patched mindtorch.Tensor.random_ for CUDA compatibility")
            
            # Patch torch.arange to always use CPU device
            import mindtorch as torch
            original_arange = torch.arange
            
            def patched_arange(*args, **kwargs):
                """强制所有 arange 调用使用 CPU 设备，避免 device:3 路由错误"""
                # 如果已经指定了device，保持不变；否则强制使用CPU
                if 'device' not in kwargs:
                    kwargs['device'] = 'cpu'
                return original_arange(*args, **kwargs)
            
            torch.arange = patched_arange
            print("✓ Patched torch.arange to always use CPU device")
            
            # Patch torch.split to handle mindtorch edge cases
            original_split = torch.split
            
            def patched_split(tensor, split_size_or_sections, dim=0):
                """修复 torch.split 在 mindtorch 中的兼容性问题 - 完全绕过"""
                import sys
                
                # 获取 split_size_or_sections 的实际类型和值
                if hasattr(split_size_or_sections, 'tolist'):
                    split_list = split_size_or_sections.tolist()
                else:
                    split_list = split_size_or_sections
                
                # 强制调试输出
                list_len = len(split_list) if isinstance(split_list, list) else 0
                if list_len > 50:
                    print(f"⚠️  SPLIT DEBUG: list length={list_len}, first 5 items={split_list[:5] if isinstance(split_list, list) else 'N/A'}", file=sys.stderr)
                    print(f"⚠️  SPLIT DEBUG: tensor.shape={tensor.shape}, dim={dim}", file=sys.stderr)
                    
                    # 强制使用手动切片，完全绕过 mindtorch split
                    if isinstance(split_list, list) and len(set(split_list[:min(10, len(split_list))])) == 1:
                        chunk_size = split_list[0]
                        dim_size = tensor.size(dim)
                        num_chunks = dim_size // chunk_size
                        
                        print(f"⚠️  FORCING manual split: {num_chunks} chunks of size {chunk_size}", file=sys.stderr)
                        
                        # 手动切片
                        result = []
                        for i in range(num_chunks):
                            start_idx = i * chunk_size
                            end_idx = start_idx + chunk_size
                            
                            # 使用 narrow 而不是索引切片（更兼容）
                            chunk = tensor.narrow(dim, start_idx, chunk_size)
                            result.append(chunk)
                        
                        print(f"✓ Manual split succeeded: {len(result)} chunks", file=sys.stderr)
                        return result
                
                # 正常路径
                return original_split(tensor, split_list if isinstance(split_list, (list, tuple)) else split_size_or_sections, dim)
            
            torch.split = patched_split
            print("✓ Patched torch.split with forced debug and manual chunking")
            
        except Exception as e:
            print(f"⚠️  Failed to patch operations: {e}")
    except Exception as e:
        print(f"⚠️  Failed to set CPU-only mode: {e}")

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from .dataset import OCRDataset, OCRDataCollator, split_dataset


# 自定义Trainer：CPU offload模式下禁止移动base model
class CPUOffloadTrainer(Trainer):
    """
    Custom Trainer for CPU offload mode.
    Prevents moving the entire model to NPU during initialization and training.
    """
    def __init__(self, *args, cpu_offload=False, **kwargs):
        self.cpu_offload = cpu_offload
        self.cpu_generator = None
        if cpu_offload:
            # 创建CPU generator以修复mindtorch DataLoader的inplace_random问题
            self.cpu_generator = torch.Generator(device='cpu')
            self.cpu_generator.manual_seed(42)
            logger.info("✓ Created CPU generator for DataLoader in Trainer")
        super().__init__(*args, **kwargs)
    
    def _move_model_to_device(self, model, device):
        """Override to prevent moving base model in CPU offload mode"""
        if self.cpu_offload:
            # CPU offload模式：不移动模型（LoRA参数已在NPU，base model保持在CPU）
            logger.info("⚠️ CPU offload: Skipping model.to(device) - base model stays on CPU")
            return
        else:
            # 正常模式：移动整个模型
            super()._move_model_to_device(model, device)
    
    def _inner_training_loop(self, *args, **kwargs):
        """Override to prevent accelerate from moving model during training"""
        if self.cpu_offload:
            # CPU offload模式：monkey patch accelerate.prepare_model以防止移动模型
            original_prepare_model = self.accelerator.prepare_model
            
            def patched_prepare_model(model, device_placement=None):
                """Patched version that skips model.to(device)"""
                logger.info("⚠️ CPU offload: Accelerate.prepare_model patched - keeping model on CPU")
                # 直接返回模型，不进行任何设备移动
                return model
            
            self.accelerator.prepare_model = patched_prepare_model
            
            try:
                return super()._inner_training_loop(*args, **kwargs)
            finally:
                # 恢复原始方法
                self.accelerator.prepare_model = original_prepare_model
        else:
            return super()._inner_training_loop(*args, **kwargs)
    
    def get_train_dataloader(self):
        """Override to use SequentialSampler in CPU offload mode to avoid mindtorch randperm issues"""
        if self.cpu_offload:
            # CPU offload模式：强制使用SequentialSampler避免RandomSampler调用randperm
            from torch.utils.data import DataLoader, SequentialSampler
            
            train_dataset = self.train_dataset
            data_collator = self.data_collator
            
            # 显式创建SequentialSampler（不调用任何随机函数）
            sampler = SequentialSampler(train_dataset)
            
            # 手动创建DataLoader
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=sampler,
                collate_fn=data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
            
            logger.info("⚠️ CPU offload: Using SequentialSampler to avoid randperm on CUDA")
            return train_dataloader
        else:
            return super().get_train_dataloader()
    
    def get_eval_dataloader(self, eval_dataset=None):
        """Override to use SequentialSampler for eval in CPU offload mode"""
        if self.cpu_offload:
            from torch.utils.data import DataLoader, SequentialSampler
            
            if eval_dataset is None:
                eval_dataset = self.eval_dataset
            
            sampler = SequentialSampler(eval_dataset)
            
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
            
            logger.info("⚠️ CPU offload: Using SequentialSampler for eval")
            return eval_dataloader
        else:
            return super().get_eval_dataloader(eval_dataset)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None,
) -> LoraConfig:
    """
    创建 LoRA 配置
    
    Args:
        r: LoRA rank, 推荐 8-64
        lora_alpha: LoRA alpha, 推荐 16-128
        lora_dropout: Dropout 比例
        target_modules: 目标模块列表
        
    Returns:
        LoraConfig 实例
    """
    if target_modules is None:
        # Qwen2-VL 默认目标模块
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",     # FFN
        ]
    
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    logger.info(f"LoRA config created: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"Target modules: {target_modules}")
    
    return config


def train_lora(
    model_name_or_path: str,
    data_path: str,
    output_dir: str,
    image_folder: Optional[str] = None,
    # LoRA 参数
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    # 训练参数
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 2048,
    # 其他参数
    save_steps: int = 100,
    logging_steps: int = 10,
    eval_steps: int = 100,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    fp16: bool = True,
    gradient_checkpointing: bool = True,
    # 数据集分割
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    # 分布式训练
    local_rank: int = -1,
    # CPU offload（用于节省NPU内存）
    cpu_offload: bool = False,
):
    """
    执行 LoRA 微调
    
    Args:
        model_name_or_path: 预训练模型路径或名称
        data_path: 训练数据 JSON 文件路径
        output_dir: 输出目录
        image_folder: 图片文件夹路径
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        num_epochs: 训练轮数
        batch_size: 批大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        max_length: 最大序列长度
        save_steps: 保存步数
        logging_steps: 日志步数
        eval_steps: 评估步数
        warmup_ratio: Warmup 比例
        weight_decay: 权重衰减
        fp16: 是否使用 FP16
        gradient_checkpointing: 是否使用梯度检查点
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
    """
    # 检测分布式环境 - Trainer会自动处理初始化
    is_distributed = local_rank != -1 or int(os.environ.get('RANK', -1)) != -1
    
    # 获取rank信息（可能由环境变量设置）
    if is_distributed:
        rank = int(os.environ.get('RANK', local_rank if local_rank != -1 else 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        # 注意：不要在这里手动init_process_group，Trainer会自动处理
    else:
        rank = 0
        world_size = 1
    
    # 设置随机种子
    torch.manual_seed(random_seed)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 只在主进程打印信息
    if rank == 0:
        logger.info("=" * 80)
        logger.info("Starting LoRA Fine-tuning")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name_or_path}")
        logger.info(f"Data: {data_path}")
        logger.info(f"Output: {output_dir}")
    logger.info(f"LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"Training: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
    
    if rank == 0:
        logger.info(f"Distributed: rank={rank}/{world_size}" if is_distributed else "Single device training")
    
    # 1. 加载 processor
    if rank == 0:
        logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    
    # 2. 加载数据集
    if rank == 0:
        logger.info("Loading dataset...")
    full_dataset = OCRDataset(
        data_path=data_path,
        processor=processor,
        max_length=max_length,
        image_folder=image_folder,
    )
    
    # 分割数据集
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )
    
    # 3. 加载模型
    if rank == 0:
        logger.info("Loading model...")
        if cpu_offload:
            logger.info("Using CPU offload for optimizer states and gradients")
    
    # 配置设备
    if is_distributed:
        # DDP模式：每个进程使用指定NPU
        device_id = local_rank if local_rank != -1 else int(os.environ.get('LOCAL_RANK', 0))
        device = f'npu:{device_id}' if hasattr(torch, 'npu') else f'cuda:{device_id}'
        if rank == 0:
            logger.warning("⚠️ DDP mode requires ~14GB per NPU. Recommend single NPU training.")
    else:
        # 单NPU模式
        device = 'npu:0' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cuda:0'
    
    # CPU offload模式下，NPU已在模块导入前隐藏，现在加载模型和LoRA
    if cpu_offload and not is_distributed:
        if rank == 0:
            logger.info("Step 1: Loading model with NPU hidden (set at startup)...")
    
    try:
        # 加载模型到CPU（NPU已被隐藏）
        if rank == 0:
            logger.info("Loading base model to CPU...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if fp16 else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if rank == 0:
            logger.info("✓ Base model loaded")
        
        # Patch Qwen2VL VisionAttention to avoid mindtorch split issues
        try:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionAttention
            original_vision_attn_forward = Qwen2VLVisionAttention.forward
            
            def patched_vision_attn_forward(self, hidden_states, cu_seqlens, rotary_pos_emb):
                """绕过 mindtorch split 问题的 vision attention forward"""
                import torch
                
                seq_length = hidden_states.shape[0]
                q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
                
                # 跳过 multi-resolution attention 的 split 操作
                # 直接使用标准 attention 而不是分割头
                q = self.core_attention_flash(q, k, v, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
                
                context_layer = q.reshape(seq_length, -1)
                attn_output = self.proj(context_layer)
                return attn_output
            
            Qwen2VLVisionAttention.forward = patched_vision_attn_forward
            if rank == 0:
                logger.info("✓ Patched Qwen2VL VisionAttention to bypass split issues")
        except Exception as e:
            if rank == 0:
                logger.warning(f"⚠️  Failed to patch VisionAttention: {e}")
        
        # 启用梯度检查点
        if gradient_checkpointing:
            # 暂时禁用 - mindtorch split 操作在 gradient checkpointing 中有兼容性问题
            # model.gradient_checkpointing_enable()
            if rank == 0:
                logger.info("⚠️ Gradient checkpointing disabled due to mindtorch compatibility")
                logger.info("   (torch.split issue in Qwen2VL attention with checkpointing)")
        
        # 启用输入梯度
        model.enable_input_require_grads()
        
        # 4. 创建 LoRA 模型（仍在CPU offload模式，NPU隐藏中）
        if rank == 0:
            logger.info("Step 2: Applying LoRA configuration (NPU still hidden)...")
        lora_config = create_lora_config(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
        model = get_peft_model(model, lora_config)
    
    finally:
        # 恢复NPU可见性（使用启动时保存的原始值）
        if cpu_offload and not is_distributed:
            if '_original_npu_devices' in globals():
                if _original_npu_devices is not None:
                    os.environ['NPU_VISIBLE_DEVICES'] = _original_npu_devices
                else:
                    # 原本没有设置，现在设置为0（单NPU）
                    os.environ['NPU_VISIBLE_DEVICES'] = '0'
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            
            # 恢复mindspore和mindtorch为NPU模式（用于后续训练）
            try:
                import mindspore
                # 关键：切换mindspore context回Ascend
                mindspore.set_context(device_target='Ascend', device_id=0)
                if rank == 0:
                    logger.info(f"✓ MindSpore context restored to Ascend (device_id=0)")
                
                from mindtorch._bind import set_default_device
                set_default_device(device)
                if rank == 0:
                    logger.info(f"✓ mindtorch default device restored to {device}")
            except Exception as e:
                if rank == 0:
                    logger.warning(f"⚠️  Failed to restore NPU mode: {e}")
            
            if rank == 0:
                logger.info(f"✓ NPU device restored (NPU_VISIBLE_DEVICES={os.environ.get('NPU_VISIBLE_DEVICES')})")
                logger.info("✓ Model and LoRA successfully loaded on CPU")
    
    # 移动LoRA参数到NPU（基础模型保持在CPU）
    if cpu_offload and not is_distributed:
        if rank == 0:
            logger.info(f"Step 3: Moving LoRA parameters to {device}...")
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:  # 只移动可训练参数（LoRA）
                param.data = param.data.to(device)
                trainable_params.append(name)
        if rank == 0:
            logger.info(f"✓ Moved {len(trainable_params)} LoRA parameters to {device}")
            logger.info(f"✓ Base model remains on CPU (saves ~14GB NPU memory)")
    else:
        # 非offload模式：整个模型移动到device
        if rank == 0:
            logger.info(f"Moving entire model to {device}...")
        model = model.to(device)
        if rank == 0:
            logger.info(f"✓ Model on {device}")
    
    # 清理内存
    gc.collect()
    if hasattr(torch, 'npu'):
        torch.npu.empty_cache()
    
    if rank == 0:
        model.print_trainable_parameters()
    
    # 5. 创建数据整理器
    data_collator = OCRDataCollator(
        processor=processor,
        padding=True,
        max_length=max_length,
    )
    
    # 6. 配置训练参数
    # 单进程模式：需要mock distributed以避免accelerate初始化process group
    if not is_distributed:
        # 临时mock torch.distributed函数，骗过accelerate的分布式检测
        original_is_available = torch.distributed.is_available
        original_is_initialized = torch.distributed.is_initialized
        original_get_world_size = torch.distributed.get_world_size
        original_get_rank = torch.distributed.get_rank
        
        torch.distributed.is_available = lambda: False
        torch.distributed.is_initialized = lambda: False
        torch.distributed.get_world_size = lambda *args, **kwargs: 1
        torch.distributed.get_rank = lambda *args, **kwargs: 0
    
    try:
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=3,
        fp16=fp16,
        dataloader_num_workers=0 if cpu_offload else 2,  # CPU offload模式禁用多进程
        remove_unused_columns=False,
        eval_strategy="steps",  # transformers 4.57+ 使用eval_strategy替代evaluation_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        seed=random_seed,
        # 分布式训练配置
        local_rank=local_rank,
        ddp_find_unused_parameters=False,  # 优化DDP性能
        ddp_backend="hccl" if hasattr(torch, 'npu') and torch.npu.is_available() else None,
        )
    finally:
        # 恢复原始distributed函数
        if not is_distributed:
            torch.distributed.is_available = original_is_available
            torch.distributed.is_initialized = original_is_initialized
            torch.distributed.get_world_size = original_get_world_size
            torch.distributed.get_rank = original_get_rank
    
    if rank == 0 and cpu_offload:
        logger.info("⚠️ CPU offload mode: Training will be slower but use less NPU memory")
    
    # 7. 创建 Trainer（CPU offload模式使用自定义Trainer）
    trainer = CPUOffloadTrainer(
        model=model,
        cpu_offload=cpu_offload,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 8. 开始训练
    if rank == 0:
        logger.info("Starting training...")
    trainer.train()
    
    # 9. 保存最终模型（Trainer会自动处理分布式保存）
    final_output_dir = output_dir / "final_model"
    trainer.save_model(str(final_output_dir))
    if rank == 0:
        processor.save_pretrained(str(final_output_dir))
        logger.info(f"Model saved to {final_output_dir}")
        logger.info("Training completed!")
    
    return model, processor


def load_lora_model(
    base_model_path: str,
    lora_path: str,
    device: str = None,
):
    """
    加载 LoRA 微调后的模型
    
    Args:
        base_model_path: 基础模型路径
        lora_path: LoRA 权重路径
        device: 设备 (None表示自动检测)
        
    Returns:
        (model, processor)
    """
    # 自动检测设备
    if device is None:
        try:
            import torch_npu
            if torch.npu.is_available():
                device = "npu:0"
                logger.info("Detected NPU device")
        except ImportError:
            pass
        
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Detected CUDA device")
            else:
                device = "cpu"
                logger.info("Using CPU device")
    
    logger.info(f"Using device: {device}")
    logger.info(f"Loading base model from {base_model_path}")
    
    # 加载 processor
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    # 加载基础模型 (禁用device_map避免meta tensor问题)
    logger.info("Loading base model...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,  # 禁用低内存模式,确保所有权重都加载
        attn_implementation="eager",  # 使用传统attention实现,NPU不支持SDPA
        trust_remote_code=True
    )
    
    # 加载 LoRA 权重
    logger.info(f"Loading LoRA weights from {lora_path}")
    from pathlib import Path
    import numpy as np
    import os
    
    lora_path = Path(lora_path).resolve()
    
    # 检查是否是.npz格式的权重（MindSpore格式）
    adapter_file = lora_path / "adapter_model.npz"
    if adapter_file.exists():
        logger.info("Loading MindSpore format (.npz) LoRA weights")
        # 为基础模型添加LoRA配置
        from peft import LoraConfig, get_peft_model
        import json
        
        # 读取adapter配置
        config_file = lora_path / "adapter_config.json"
        with open(config_file) as f:
            adapter_config = json.load(f)
        
        # 创建LoRA配置
        lora_config = LoraConfig(
            r=adapter_config.get("r", 8),
            lora_alpha=adapter_config.get("lora_alpha", 16),
            target_modules=adapter_config.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=adapter_config.get("lora_dropout", 0.1),
            bias=adapter_config.get("bias", "none"),
            task_type=adapter_config.get("task_type", "CAUSAL_LM"),
        )
        
        # 将基础模型转换为PEFT模型(在CPU上)
        logger.info("Creating PEFT model...")
        model = get_peft_model(base_model, lora_config)
        
        # 加载.npz权重
        logger.info("Loading weights from .npz file...")
        weights = np.load(str(adapter_file))
        state_dict = {}
        for key in weights.files:
            # 转换numpy数组为torch tensor
            tensor = torch.from_numpy(weights[key])
            state_dict[key] = tensor
        
        # 加载权重到模型(使用assign=True直接替换meta tensors)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
        logger.info(f"Loaded {len(state_dict)} weight tensors")
        if missing_keys:
            logger.warning(f"Missing keys: {len(missing_keys)} keys not found in checkpoint")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys in checkpoint not used")
    else:
        # 标准PEFT格式
        logger.info("Loading standard PEFT format LoRA weights")
        model = PeftModel.from_pretrained(
            base_model, 
            str(lora_path),
            is_trainable=False
        )
    
    # 现在移动到目标设备
    logger.info(f"Moving model to device: {device}")
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    return model, processor


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL LoRA Fine-tuning")
    
    # 基础参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="预训练模型路径或名称")
    parser.add_argument("--data_path", type=str, required=True,
                        help="训练数据 JSON 文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--image_folder", type=str, default=None,
                        help="图片文件夹路径")
    
    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (推荐 8-64)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (推荐 16-128)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="最大序列长度")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="保存步数")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志步数")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="评估步数")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup 比例")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--no_fp16", action="store_true",
                        help="不使用 FP16")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="不使用梯度检查点")
    
    # 数据集分割
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="测试集比例")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="随机种子")
    
    # 分布式训练参数（torch.distributed.launch 会自动添加）
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="分布式训练的本地 rank（由 torch.distributed.launch 自动设置）")
    parser.add_argument("--local-rank", type=int, default=-1,
                        help="分布式训练的本地 rank（由 torch.distributed.launch 自动设置，兼容格式）")
    
    # 内存优化选项
    parser.add_argument("--cpu_offload", action="store_true",
                        help="将部分模型层offload到CPU以节省NPU内存")
    
    args = parser.parse_args()
    
    # 处理local_rank参数（支持下划线和连字符两种格式）
    local_rank = max(args.local_rank, getattr(args, 'local-rank', -1))
    
    # 执行训练
    train_lora(
        model_name_or_path=args.model_name_or_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        image_folder=args.image_folder,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=not args.no_fp16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        local_rank=local_rank,
        cpu_offload=args.cpu_offload,
    )


if __name__ == "__main__":
    main()
