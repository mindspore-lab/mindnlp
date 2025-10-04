import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import mindspore 
from mindspore import context as ms_context
import mindnlp

import datasets
from mindnlp.transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_last_checkpoint
)

from mind_openr1.configs import ScriptArguments
from mind_openr1.sft_trainer import SFTTrainer, SFTConfig
from mind_openr1.utils import get_dataset
from mind_openr1.utils.callbacks import get_callbacks

ms_context.set_context(mode=ms_context.PYNATIVE_MODE)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration compatible with mindnlp"""
    model_name_or_path: str
    model_revision: str = "main"
    trust_remote_code: bool = False
    use_flash_attention_2: bool = False
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    lora_target_modules: Optional[list] = None
    use_peft: bool = False


def get_tokenizer_mindnlp(model_args: ModelConfig):
    """Get tokenizer using mindnlp"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer


def get_model_mindnlp(model_args: ModelConfig):
    """Get model using mindnlp"""
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        ms_dtype=mindspore.float16 if model_args.use_flash_attention_2 else mindspore.float32,
    )
    
    return model


def get_peft_config_dict(model_args: ModelConfig):
    """Get PEFT configuration if enabled"""
    if not model_args.use_peft:
        return None
        
    peft_config = {
        "r": model_args.lora_r or 16,
        "lora_alpha": model_args.lora_alpha or 32,
        "lora_dropout": model_args.lora_dropout or 0.1,
        "target_modules": model_args.lora_target_modules or ["q_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    
    return peft_config


def setup_chat_format(model, tokenizer):
    """Setup chat format for model and tokenizer"""
    if tokenizer.chat_template is None:
        logger.info("No chat template provided, setting up ChatML format")
        # Simple ChatML template
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        
        # Add special tokens if needed
        special_tokens = {
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]
        }
        tokenizer.add_special_tokens(special_tokens)
        
        # Resize model embeddings
        model.resize_token_embeddings(len(tokenizer))
        
    return model, tokenizer


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO if training_args.logging_steps > 0 else logging.WARNING
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    ######################################
    # Load dataset, tokenizer, and model #
    ######################################
    dataset = get_dataset(script_args)
    
    # Optionally truncate training split if max_train_samples is provided
    if getattr(script_args, "max_train_samples", None):
        train_split = script_args.dataset_train_split
        max_n = int(script_args.max_train_samples)
        if max_n > 0:
            dataset[train_split] = dataset[train_split].select(range(min(max_n, len(dataset[train_split]))))
            
    tokenizer = get_tokenizer_mindnlp(model_args)
    model = get_model_mindnlp(model_args)

    # Setup chat format if needed
    model, tokenizer = setup_chat_format(model, tokenizer)

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        processing_class=tokenizer,
        peft_config=get_peft_config_dict(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    metrics = train_result
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    
    # Save model
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1", "mindspore", "mindnlp"],
    }
    
    # Create model card
    trainer.create_model_card(**kwargs)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # Script arguments
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_train_split", type=str, default="train")
    parser.add_argument("--dataset_test_split", type=str, default="test")
    parser.add_argument("--max_train_samples", type=int, default=None)
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_revision", type=str, default="main")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_flash_attention_2", action="store_true")
    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None)
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--eval_strategy", type=str, default="steps")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--dataset_text_field", type=str, default="text")
    
    args = parser.parse_args()
    
    # Create config objects
    script_args = ScriptArguments(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_train_split=args.dataset_train_split,
        dataset_test_split=args.dataset_test_split,
        max_train_samples=args.max_train_samples,
    )
    
    model_args = ModelConfig(
        model_name_or_path=args.model_name_or_path,
        model_revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        use_flash_attention_2=args.use_flash_attention_2,
        use_peft=args.use_peft,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )
    
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_seq_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        max_steps=args.max_steps,
        seed=args.seed,
        do_eval=args.do_eval,
        push_to_hub=args.push_to_hub,
        resume_from_checkpoint=args.resume_from_checkpoint,
        dataset_text_field=args.dataset_text_field,
    )
    
    main(script_args, training_args, model_args)