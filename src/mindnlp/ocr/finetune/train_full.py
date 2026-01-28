"""
Qwen2-VL Full-Parameter Fine-tuning Script
全参数微调实现 (适用于大规模数据集 100K+)
"""
import os
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    get_scheduler,
)
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .dataset import OCRDataset, OCRDataCollator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """初始化分布式训练环境"""
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        return local_rank, dist.get_rank(), dist.get_world_size()
    return -1, 0, 1


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_full(
    model_name_or_path: str,
    data_path: str,
    output_dir: str,
    image_folder: Optional[str] = None,
    # 训练参数
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-5,
    max_length: int = 2048,
    save_steps: int = 100,
    logging_steps: int = 10,
    eval_steps: int = 100,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    # 优化选项
    fp16: bool = True,
    bf16: bool = False,
    gradient_checkpointing: bool = True,
    # 数据集分割
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    # 分布式训练
    local_rank: int = -1,
):
    """
    全参数微调主函数

    Args:
        model_name_or_path: 预训练模型路径
        data_path: 训练数据路径
        output_dir: 输出目录
        image_folder: 图片文件夹
        num_epochs: 训练轮数
        batch_size: 每个GPU的批大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率 (推荐 1e-5 ~ 5e-5)
        max_length: 最大序列长度
        save_steps: 保存检查点步数
        logging_steps: 日志记录步数
        eval_steps: 评估步数
        warmup_ratio: Warmup 比例
        weight_decay: 权重衰减
        max_grad_norm: 梯度裁剪阈值
        fp16: 使用 FP16 混合精度
        bf16: 使用 BF16 混合精度
        gradient_checkpointing: 使用梯度检查点
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
        local_rank: 分布式训练 local rank
    """

    # 设置随机种子
    torch.manual_seed(random_seed)

    # 初始化分布式训练
    if local_rank == -1:
        local_rank, rank, world_size = setup_distributed()
    else:
        rank = local_rank
        world_size = int(os.environ.get('WORLD_SIZE', 1))

    is_distributed = world_size > 1
    is_main_process = rank == 0

    if is_main_process:
        logger.info("=" * 80)
        logger.info("Qwen2-VL Full-Parameter Fine-tuning")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name_or_path}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Distributed: {is_distributed} (World size: {world_size})")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Batch size per GPU: {batch_size}")
        logger.info(f"Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps * world_size}")
        logger.info("=" * 80)

    # 创建输出目录
    output_path = Path(output_dir)
    if is_main_process:
        output_path.mkdir(parents=True, exist_ok=True)

    # 加载 Processor
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )

    # 加载模型
    logger.info("Loading model...")

    # 确定设备和数据类型
    device = f"cuda:{local_rank}" if local_rank != -1 else "cuda"

    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map={"": device} if not is_distributed else None,
        trust_remote_code=True,
    )

    # 启用梯度检查点
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # 包装为 DDP
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        logger.info(f"Model wrapped with DDP on device {local_rank}")
    else:
        model = model.to(device)

    # 准备数据集
    logger.info("Loading dataset...")

    # 加载数据
    dataset = OCRDataset(
        data_path=data_path,
        processor=processor,
        image_folder=image_folder,
        max_length=max_length,
    )

    # 分割数据集
    train_dataset, val_dataset, test_dataset = dataset.split_dataset(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # 创建数据加载器
    data_collator = OCRDataCollator(processor=processor)

    train_sampler = None
    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=random_seed,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True,
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # 学习率调度器
    num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    logger.info(f"Training steps: {num_training_steps}")
    logger.info(f"Warmup steps: {num_warmup_steps}")

    # 混合精度训练
    scaler = None
    if fp16:
        scaler = torch.cuda.amp.GradScaler()
        logger.info("FP16 mixed precision enabled")

    # 训练循环
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not is_main_process,
        )

        for step, batch in enumerate(progress_bar):
            # 移动数据到设备
            batch = {k: v.to(device) for k, v in batch.items()}

            # 前向传播
            if fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
            else:
                outputs = model(**batch)
                loss = outputs.loss

            # 梯度累积
            loss = loss / gradient_accumulation_steps

            # 反向传播
            if fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()

            # 梯度更新
            if (step + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪
                if fp16:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # 优化器步进
                if fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # 日志
                if global_step % logging_steps == 0 and is_main_process:
                    avg_loss = total_loss / logging_steps
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'
                    })
                    total_loss = 0

                # 评估
                if global_step % eval_steps == 0 and is_main_process:
                    val_loss = evaluate(model, val_loader, device, fp16)
                    logger.info(f"Step {global_step} - Val Loss: {val_loss:.4f}")

                    # 保存最佳模型
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_path = output_path / "best_model"
                        save_model(model, processor, save_path, is_distributed)
                        logger.info(f"Best model saved to {save_path}")

                    model.train()

                # 保存检查点
                if global_step % save_steps == 0 and is_main_process:
                    checkpoint_path = output_path / f"checkpoint-{global_step}"
                    save_model(model, processor, checkpoint_path, is_distributed)
                    logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Epoch 结束
        if is_main_process:
            logger.info(f"Epoch {epoch + 1} completed")

    # 保存最终模型
    if is_main_process:
        final_path = output_path / "final_model"
        save_model(model, processor, final_path, is_distributed)
        logger.info(f"Final model saved to {final_path}")

    # 清理
    cleanup_distributed()

    logger.info("Training completed!")

    return model, processor


def evaluate(model, val_loader, device, fp16=False):
    """评估模型"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            if fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
            else:
                outputs = model(**batch)

            total_loss += outputs.loss.item()
            num_batches += 1

    return total_loss / num_batches


def save_model(model, processor, save_path, is_distributed=False):
    """保存模型"""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # 保存模型
    if is_distributed:
        model.module.save_pretrained(save_path)
    else:
        model.save_pretrained(save_path)

    # 保存 processor
    processor.save_pretrained(save_path)


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL Full-Parameter Fine-tuning")

    # 基础参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="预训练模型路径或名称")
    parser.add_argument("--data_path", type=str, required=True,
                        help="训练数据 JSON 文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--image_folder", type=str, default=None,
                        help="图片文件夹路径")

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="每个GPU的批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="学习率 (推荐 1e-5 ~ 5e-5)")
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
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪阈值")

    # 优化选项
    parser.add_argument("--fp16", action="store_true",
                        help="使用 FP16 混合精度")
    parser.add_argument("--bf16", action="store_true",
                        help="使用 BF16 混合精度")
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

    # 分布式训练参数
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="分布式训练的本地 rank")

    args = parser.parse_args()

    # 执行训练
    train_full(
        model_name_or_path=args.model_name_or_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        image_folder=args.image_folder,
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
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        local_rank=args.local_rank,
    )


if __name__ == "__main__":
    main()
