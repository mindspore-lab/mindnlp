"""
OCR Toolkit - 统一的训练、推理和评估工具
支持：
1. 训练 LoRA 模型（使用预提取的视觉特征）
2. 单图片推理
3. 批量推理和评估
"""
import os
import sys
import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# 添加 src 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from mindnlp.ocr.finetune.dataset_preextracted import PreExtractedOCRDataset, PreExtractedDataCollator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 训练相关
# ============================================================================

def train_with_preextracted(
    model_name_or_path: str,
    data_path: str,
    features_dir: str,
    output_dir: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 2e-4,
    max_length: int = 256,
    device: str = "npu:0",
    logging_steps: int = 10,
    save_steps: int = 500,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01
):
    """使用预提取的视觉特征训练 LoRA 模型"""
    logger.info("="*80)
    logger.info("Starting LoRA Fine-tuning with Pre-extracted Features")
    logger.info("="*80)
    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Features: {features_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"Training: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
    logger.info(f"Device: {device}")
    
    # 1. 加载 processor
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    
    # 2. 加载基础模型
    logger.info("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    
    # 3. 配置 LoRA
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=['gate_proj', 'q_proj', 'up_proj', 'o_proj', 'v_proj', 'k_proj', 'down_proj']
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. 创建数据集
    logger.info("Creating dataset...")
    train_dataset = PreExtractedOCRDataset(
        data_path=data_path,
        processor=processor,
        features_dir=features_dir,
        max_length=max_length
    )
    
    data_collator = PreExtractedDataCollator(processor=processor)
    
    # 5. 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=False,
        remove_unused_columns=False,
        report_to="tensorboard",
        skip_memory_metrics=True,
        optim="sgd",
        save_safetensors=False,
        save_only_model=True,
    )
    
    # 6. 自定义 Trainer
    class CustomTrainer(Trainer):
        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            for key in ['input_ids', 'attention_mask', 'labels']:
                if key in inputs and isinstance(inputs[key], (list, tuple)):
                    inputs[key] = torch.tensor(inputs[key], dtype=torch.long).to(self.args.device)
            return inputs
        
        def floating_point_ops(self, inputs):
            return 0
        
        def _save(self, output_dir: Optional[str] = None, state_dict=None):
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")
            
            if hasattr(self.model, 'save_pretrained'):
                state_dict_to_save = self.model.state_dict() if state_dict is None else state_dict
                numpy_state_dict = {}
                for key, value in state_dict_to_save.items():
                    if hasattr(value, 'cpu') and hasattr(value, 'numpy'):
                        numpy_state_dict[key] = value.cpu().numpy()
                    else:
                        numpy_state_dict[key] = value
                
                import numpy as np
                weights_file = os.path.join(output_dir, "adapter_model.npz")
                np.savez(weights_file, **numpy_state_dict)
                
                if hasattr(self.model, 'peft_config'):
                    for adapter_name, peft_config in self.model.peft_config.items():
                        peft_config.save_pretrained(output_dir)
                
                logger.info(f"Model weights saved to {weights_file}")
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 7. 开始训练
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("="*80)
    logger.info("Training completed!")
    logger.info("="*80)


# ============================================================================
# 推理相关
# ============================================================================

def load_lora_model(base_model_path: str, checkpoint_dir: str, device: str = "npu:0"):
    """加载 LoRA 模型"""
    logger.info(f"Loading base model from {base_model_path}...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
        local_files_only=True
    )
    
    logger.info(f"Loading LoRA weights from {checkpoint_dir}...")
    weights_file = os.path.join(checkpoint_dir, "adapter_model.npz")
    numpy_weights = np.load(weights_file)
    state_dict = {k: torch.from_numpy(numpy_weights[k]) for k in numpy_weights.files}
    
    with open(os.path.join(checkpoint_dir, "adapter_config.json")) as f:
        peft_config = json.load(f)
    
    lora_config = LoraConfig(**peft_config)
    model = get_peft_model(base_model, lora_config)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    logger.info("✅ Model loaded successfully")
    return model


def inference_single(
    model_path: str,
    checkpoint_dir: str,
    image_path: str,
    question: Optional[str] = None,
    device: str = "npu:0"
):
    """单张图片推理"""
    logger.info("="*80)
    logger.info("Single Image Inference")
    logger.info("="*80)
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = load_lora_model(model_path, checkpoint_dir, device)
    
    image = Image.open(image_path).convert('RGB')
    if question is None:
        question = "Extract all text from this document, including labels and values. Format as JSON with keys for each field."
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]
    }]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    logger.info("Generating...")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    answer = result.split("assistant\n")[-1] if "assistant\n" in result else result
    
    logger.info("="*80)
    logger.info("Result:")
    logger.info("="*80)
    print(answer)
    
    return answer


def inference_batch(
    model_path: str,
    checkpoint_dir: str,
    test_data_path: str,
    output_file: str,
    device: str = "npu:0"
):
    """批量推理"""
    logger.info("="*80)
    logger.info("Batch Inference")
    logger.info("="*80)
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    logger.info(f"Loaded {len(test_data)} test samples")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = load_lora_model(model_path, checkpoint_dir, device)
    
    results = []
    for idx, sample in enumerate(test_data):
        logger.info(f"Processing {idx+1}/{len(test_data)}: {sample['image']}")
        
        try:
            image = Image.open(sample['image']).convert('RGB')
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract all text from this document."}
                ]
            }]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt")
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            
            result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            answer = result.split("assistant\n")[-1] if "assistant\n" in result else result
            
            results.append({
                "image": sample['image'],
                "ground_truth": sample['conversations'][1]['value'],
                "prediction": answer
            })
        except Exception as e:
            logger.error(f"Error: {e}")
            results.append({
                "image": sample['image'],
                "ground_truth": sample['conversations'][1]['value'],
                "prediction": f"ERROR: {str(e)}"
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Results saved to {output_file}")
    return results


# ============================================================================
# 评估相关
# ============================================================================

def evaluate(predictions_file: str, output_file: Optional[str] = None):
    """评估预测结果"""
    logger.info("="*80)
    logger.info("Model Evaluation")
    logger.info("="*80)
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    logger.info(f"Loaded {len(predictions)} predictions")
    
    # 计算指标
    exact_match = sum(1 for p in predictions if p['prediction'].strip() == p['ground_truth'].strip())
    exact_match_rate = exact_match / len(predictions)
    
    # Token级准确率
    total_tokens = 0
    correct_tokens = 0
    for pred in predictions:
        gt_tokens = set(pred['ground_truth'].split())
        pred_tokens = set(pred['prediction'].split())
        correct_tokens += len(gt_tokens & pred_tokens)
        total_tokens += max(len(gt_tokens), len(pred_tokens))
    
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    
    logger.info(f"📊 Exact Match: {exact_match_rate:.2%} ({exact_match}/{len(predictions)})")
    logger.info(f"📊 Token Accuracy: {token_accuracy:.2%}")
    
    results = {
        "exact_match": exact_match_rate,
        "token_accuracy": token_accuracy,
        "total_samples": len(predictions),
        "correct_samples": exact_match
    }
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✅ Results saved to {output_file}")
    
    return results


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="OCR Toolkit - Train, Inference, and Evaluate")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='Train LoRA model')
    train_parser.add_argument('--model_name_or_path', type=str, required=True)
    train_parser.add_argument('--data_path', type=str, required=True)
    train_parser.add_argument('--features_dir', type=str, required=True)
    train_parser.add_argument('--output_dir', type=str, required=True)
    train_parser.add_argument('--lora_r', type=int, default=8)
    train_parser.add_argument('--lora_alpha', type=int, default=16)
    train_parser.add_argument('--lora_dropout', type=float, default=0.1)
    train_parser.add_argument('--num_epochs', type=int, default=3)
    train_parser.add_argument('--batch_size', type=int, default=1)
    train_parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    train_parser.add_argument('--learning_rate', type=float, default=2e-4)
    train_parser.add_argument('--max_length', type=int, default=256)
    train_parser.add_argument('--device', type=str, default='npu:0')
    
    # 推理命令
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--model_path', type=str, required=True)
    infer_parser.add_argument('--checkpoint_dir', type=str, required=True)
    infer_parser.add_argument('--image_path', type=str, help='Single image path')
    infer_parser.add_argument('--test_data_path', type=str, help='Test dataset JSON')
    infer_parser.add_argument('--output_file', type=str, default='predictions.json')
    infer_parser.add_argument('--question', type=str, default=None)
    infer_parser.add_argument('--device', type=str, default='npu:0')
    
    # 评估命令
    eval_parser = subparsers.add_parser('eval', help='Evaluate predictions')
    eval_parser.add_argument('--predictions_file', type=str, required=True)
    eval_parser.add_argument('--output_file', type=str, default='evaluation.json')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_with_preextracted(
            model_name_or_path=args.model_name_or_path,
            data_path=args.data_path,
            features_dir=args.features_dir,
            output_dir=args.output_dir,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            device=args.device
        )
    elif args.command == 'infer':
        if args.image_path:
            inference_single(
                model_path=args.model_path,
                checkpoint_dir=args.checkpoint_dir,
                image_path=args.image_path,
                question=args.question,
                device=args.device
            )
        elif args.test_data_path:
            inference_batch(
                model_path=args.model_path,
                checkpoint_dir=args.checkpoint_dir,
                test_data_path=args.test_data_path,
                output_file=args.output_file,
                device=args.device
            )
        else:
            print("Error: Please specify --image_path or --test_data_path")
    elif args.command == 'eval':
        evaluate(
            predictions_file=args.predictions_file,
            output_file=args.output_file
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
