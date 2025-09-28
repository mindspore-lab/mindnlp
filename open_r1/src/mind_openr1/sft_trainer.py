"""
Supervised Fine-tuning Trainer for MindSpore/MindNLP
"""
import logging
import os
import sys
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field

import mindspore
from mindspore import nn, ops, Tensor
from mindspore.dataset import GeneratorDataset
import mindspore.context as ms_context
import mindspore.communication as comm

import datasets
from mindnlp.transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    TrainingArguments as BaseTrainingArguments
)

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig(BaseTrainingArguments):
    """
    Configuration class for SFT training specific parameters.
    Inherits from mindnlp TrainingArguments.
    """
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for input"}
    )
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "Field name containing text in the dataset"}
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Whether to pack multiple examples in a single sequence"}
    )
    dataset_train_split: str = field(
        default="train",
        metadata={"help": "Name of the training data split"}
    )
    dataset_test_split: str = field(
        default="test", 
        metadata={"help": "Name of the test data split"}
    )
    
    def __post_init__(self):
        # Ensure output directory exists
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)


class SFTTrainer:
    """
    Supervised Fine-tuning Trainer for MindSpore/MindNLP
    
    This trainer handles the training loop for supervised fine-tuning of language models.
    """
    
    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        args: Optional[SFTConfig] = None,
        train_dataset: Optional[Union[datasets.Dataset, GeneratorDataset]] = None,
        eval_dataset: Optional[Union[datasets.Dataset, GeneratorDataset]] = None,
        processing_class: Optional[PreTrainedTokenizer] = None,
        peft_config: Optional[Dict] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.model = model
        self.args = args or SFTConfig()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = processing_class
        self.peft_config = peft_config
        self.callbacks = callbacks or []
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None
        self.best_model_checkpoint = None
        
        # Setup
        self._setup_model()
        self._setup_optimizer()
        self._setup_datasets()
        
    def _setup_model(self):
        """Setup model for training"""
        if self.model is None:
            raise ValueError("Model must be provided")
            
        # Set model to training mode
        self.model.set_train(True)
        
        # Apply PEFT config if provided
        if self.peft_config:
            logger.info("Applying PEFT configuration")
            # TODO: Implement PEFT integration
            
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Get trainable parameters
        trainable_params = self.model.trainable_params()
        
        # Create optimizer
        if self.args.learning_rate is None:
            self.args.learning_rate = 5e-5
            
        self.optimizer = nn.Adam(
            trainable_params,
            learning_rate=self.args.learning_rate,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=self.args.weight_decay
        )
        
    def _setup_datasets(self):
        """Setup datasets for training"""
        if self.train_dataset is None:
            raise ValueError("Training dataset must be provided")
            
        # Process datasets if needed
        self.train_dataset = self._prepare_dataset(self.train_dataset, is_train=True)
        if self.eval_dataset is not None:
            self.eval_dataset = self._prepare_dataset(self.eval_dataset, is_train=False)
            
    def _prepare_dataset(self, dataset, is_train=True):
        """Prepare dataset for training/evaluation"""
        # If it's already a GeneratorDataset, return as is
        if isinstance(dataset, GeneratorDataset):
            return dataset
            
        # Convert HuggingFace dataset to MindSpore dataset
        def generator():
            for item in dataset:
                yield self._preprocess_function(item)
                
        column_names = ["input_ids", "attention_mask", "labels"]
        
        ms_dataset = GeneratorDataset(
            generator,
            column_names=column_names,
            shuffle=is_train
        )
        
        # Batch the dataset
        ms_dataset = ms_dataset.batch(
            batch_size=self.args.per_device_train_batch_size if is_train else self.args.per_device_eval_batch_size,
            drop_remainder=is_train
        )
        
        return ms_dataset
        
    def _preprocess_function(self, examples):
        """Preprocess a single example"""
        # Get text from the configured field
        text = examples.get(self.args.dataset_text_field, "")
        
        # Tokenize
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.args.max_seq_length,
            return_tensors="ms"
        )
        
        # For causal LM, labels are the same as input_ids
        labels = tokenized["input_ids"].copy()
        
        # Replace padding token id with -100 for loss computation
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
            
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }
        
    def compute_loss(self, model, inputs):
        """Compute loss for a batch of inputs"""
        # Forward pass
        outputs = model(**inputs)
        
        # Get loss
        if hasattr(outputs, "loss"):
            return outputs.loss
        else:
            # Compute loss manually if model doesn't return it
            logits = outputs.logits
            labels = inputs.get("labels")
            
            if labels is not None:
                # Shift for causal LM
                shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
                shift_labels = labels[..., 1:].reshape(-1)
                
                # Compute cross entropy loss
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(shift_logits, shift_labels)
                return loss
                
        return None
        
    def training_step(self, batch):
        """Perform a single training step"""
        # Convert batch to model inputs
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
        }
        
        # Forward pass and compute loss
        loss = self.compute_loss(self.model, inputs)
        
        # Backward pass
        grads = ops.grad(self.compute_loss, self.model.trainable_params())(self.model, inputs)
        
        # Update parameters
        self.optimizer(grads)
        
        return loss
        
    def train(self, resume_from_checkpoint=None):
        """Main training loop"""
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Batch size = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total optimization steps = {self.args.max_steps}")
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
            
        # Training loop
        for epoch in range(int(self.args.num_train_epochs)):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            # Iterate through batches
            for step, batch in enumerate(self.train_dataset.create_tuple_iterator()):
                loss = self.training_step(batch)
                
                epoch_loss += loss.asnumpy()
                num_batches += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % self.args.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(f"Step: {self.global_step}, Loss: {avg_loss:.4f}")
                    self.log_metrics("train", {"loss": avg_loss})
                    
                # Save checkpoint
                if self.global_step % self.args.save_steps == 0:
                    self.save_checkpoint()
                    
                # Evaluation
                if self.args.eval_strategy != "no" and self.global_step % self.args.eval_steps == 0:
                    self.evaluate()
                    
                # Check max steps
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break
                    
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.4f}")
            
            # Run callbacks
            for callback in self.callbacks:
                callback(self, epoch=epoch, loss=avg_epoch_loss)
                
            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break
                
        # Save final model
        self.save_model()
        
        return {"global_step": self.global_step}
        
    def evaluate(self):
        """Evaluation loop"""
        if self.eval_dataset is None:
            return {}
            
        logger.info("***** Running evaluation *****")
        self.model.set_train(False)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.eval_dataset.create_tuple_iterator():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2]
            }
            
            loss = self.compute_loss(self.model, inputs)
            total_loss += loss.asnumpy()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        logger.info(f"Evaluation Loss: {avg_loss:.4f}")
        
        self.model.set_train(True)
        
        metrics = {"eval_loss": avg_loss}
        self.log_metrics("eval", metrics)
        
        return metrics
        
    def save_model(self, output_dir=None):
        """Save model to disk"""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model weights
        save_checkpoint(self.model, os.path.join(output_dir, "model.ckpt"))
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            
        # Save training arguments
        with open(os.path.join(output_dir, "training_args.json"), "w") as f:
            import json
            json.dump(vars(self.args), f, indent=2)
            
        logger.info(f"Model saved to {output_dir}")
        
    def save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        self.save_model(checkpoint_dir)
        
        # Save optimizer state
        save_checkpoint(self.optimizer, os.path.join(checkpoint_dir, "optimizer.ckpt"))
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
        }
        with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w") as f:
            import json
            json.dump(state, f, indent=2)
            
    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint from disk"""
        # Load model weights
        load_checkpoint(os.path.join(checkpoint_path, "model.ckpt"), self.model)
        
        # Load optimizer state
        if os.path.exists(os.path.join(checkpoint_path, "optimizer.ckpt")):
            load_checkpoint(os.path.join(checkpoint_path, "optimizer.ckpt"), self.optimizer)
            
        # Load training state
        state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                import json
                state = json.load(f)
                self.global_step = state.get("global_step", 0)
                self.epoch = state.get("epoch", 0)
                self.best_metric = state.get("best_metric")
                
        logger.info(f"Resumed from checkpoint: {checkpoint_path}")
        
    def log_metrics(self, split, metrics):
        """Log metrics"""
        # Simple console logging
        log_str = f"[{split}] Step {self.global_step}: "
        log_str += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logger.info(log_str)
        
    def save_metrics(self, split, metrics):
        """Save metrics to file"""
        metrics_file = os.path.join(self.args.output_dir, f"{split}_metrics.json")
        with open(metrics_file, "w") as f:
            import json
            json.dump(metrics, f, indent=2)
            
    def save_state(self):
        """Save trainer state"""
        state_file = os.path.join(self.args.output_dir, "trainer_state.json")
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
        }
        with open(state_file, "w") as f:
            import json
            json.dump(state, f, indent=2)
            
    def create_model_card(self, **kwargs):
        """Create model card for the trained model"""
        # Simple model card creation
        model_card = f"""
# Model Card

## Model Details
- Model type: Causal Language Model
- Training framework: MindSpore/MindNLP
- Dataset: {kwargs.get('dataset_name', 'Unknown')}

## Training Details
- Number of epochs: {self.args.num_train_epochs}
- Batch size: {self.args.per_device_train_batch_size}
- Learning rate: {self.args.learning_rate}
- Total steps: {self.global_step}

## Tags
{kwargs.get('tags', [])}
"""
        
        with open(os.path.join(self.args.output_dir, "README.md"), "w") as f:
            f.write(model_card)
            
    def push_to_hub(self, **kwargs):
        """Push model to model hub (placeholder)"""
        logger.warning("push_to_hub is not implemented for MindSpore models yet")
        

def save_checkpoint(model, path):
    """Save model checkpoint"""
    mindspore.save_checkpoint(model, path)
    

def load_checkpoint(path, model):
    """Load model checkpoint"""
    mindspore.load_checkpoint(path, model)
