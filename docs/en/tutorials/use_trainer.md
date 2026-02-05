# Use Trainer

In the [Quick Start](./quick_start.md) tutorial, we learned how to use the Trainer API to fine-tune a model. This tutorial provides a comprehensive guide to configuring `Trainer` for optimal training outcomes.

## Overview

MindNLP patches the HuggingFace `transformers` library to work with MindSpore. This means you can use the standard HuggingFace `Trainer` and `TrainingArguments` classes directly:

```python
import mindspore
import mindnlp  # Apply patches

from transformers import Trainer, TrainingArguments
```

The `TrainingArguments` class allows you to configure essential training parameters, and `Trainer` handles the entire training loop using MindSpore as the backend.

## Configure Training Parameters

Create a `TrainingArguments` object to specify the training configuration:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50
)
```

### Basic Parameters

- **output_dir**: Directory where model checkpoints and training outputs are saved.
- **num_train_epochs**: Total number of training epochs. More epochs allow better learning but may cause overfitting.

### Optimizer Parameters

- **optim**: Optimizer type. Default is "adamw_torch" (which maps to MindSpore's AdamW).
- **learning_rate**: Initial learning rate. This is critical for convergence - too high causes instability, too low causes slow convergence.
- **weight_decay**: Regularization to prevent overfitting by penalizing large weights.
- **adam_beta1** / **adam_beta2**: Momentum parameters for Adam optimizer.
- **adam_epsilon**: Small value for numerical stability in Adam.
- **max_grad_norm**: Gradient clipping threshold to prevent exploding gradients.

### Batch Size Parameters

- **per_device_train_batch_size**: Batch size for training. Larger batches are faster but need more memory.
- **per_device_eval_batch_size**: Batch size for evaluation.
- **gradient_accumulation_steps**: Accumulate gradients over multiple steps to simulate larger batch sizes with limited memory.

### Strategy Parameters

#### Evaluation Strategy

The `eval_strategy` parameter controls when evaluation occurs:

- `"no"`: No evaluation
- `"steps"`: Evaluate every `eval_steps` training steps
- `"epoch"`: Evaluate at the end of each epoch

#### Save Strategy

The `save_strategy` parameter controls when checkpoints are saved:

- `"no"`: No saving
- `"steps"`: Save every `save_steps` training steps
- `"epoch"`: Save at the end of each epoch

#### Logging Strategy

The `logging_strategy` parameter controls when metrics are logged:

- `"no"`: No logging
- `"steps"`: Log every `logging_steps` training steps
- `"epoch"`: Log at the end of each epoch

### MindSpore-Specific Parameters

MindNLP adds support for MindSpore-specific parameters in model loading:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    ms_dtype=mindspore.float16  # Use MindSpore dtype
)
```

## Create the Trainer

Create a `Trainer` instance with your model, datasets, and configuration:

```python
import mindnlp
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

# Load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
```

### Trainer Arguments

- **model**: The model to train
- **args**: `TrainingArguments` instance with training configuration
- **train_dataset**: Training dataset
- **eval_dataset**: Evaluation dataset (optional)
- **compute_metrics**: Function to compute evaluation metrics (optional)

### Defining compute_metrics

The `compute_metrics` function computes metrics from model predictions:

```python
import evaluate
import numpy as np

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

## Start Training

Once the trainer is configured, start training:

```python
trainer.train()
```

## Complete Example

```python
import mindspore
import mindnlp
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import evaluate
import numpy as np

# Load dataset
dataset = load_dataset("imdb")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=100,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(1000)),  # Subset for demo
    eval_dataset=tokenized_datasets["test"].select(range(200)),
    compute_metrics=compute_metrics,
)

# Train
trainer.train()
```

## Advanced Features

### Mixed Precision Training

Use lower precision for faster training with less memory:

```python
training_args = TrainingArguments(
    output_dir="./output",
    fp16=True,  # Enable FP16 training
    # Or use bf16=True for bfloat16
)
```

### Gradient Checkpointing

Trade compute for memory by recomputing activations during backward pass:

```python
training_args = TrainingArguments(
    output_dir="./output",
    gradient_checkpointing=True,
)
```

### Resume from Checkpoint

Resume training from a saved checkpoint:

```python
trainer.train(resume_from_checkpoint="./results/checkpoint-500")
```

## Notes

- The Trainer automatically uses MindSpore operations through MindNLP's patching system
- All standard HuggingFace Trainer features should work
- For production training, consider using the full dataset rather than subsets
- Monitor training with TensorBoard by setting `logging_dir` in TrainingArguments
