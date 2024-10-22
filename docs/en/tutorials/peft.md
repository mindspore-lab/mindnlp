# PEFT

PEFT (Parameter-Efficient Fine-Tuning) is a technique that fine-tunes large pre-trained models with minimal parameter updates to reduce computational costs and preserve generalization. Within PEFT, LoRA ([Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)) uses low-rank matrices to efficiently adapt parts of a neural network with minimal extra parameters. This technique enables you to train large models that would typically be inaccessible on consumer devices.

In this tutorial, we'll explore this technique using MindNLP. As an example, we will use the mT0 model, which is a mT5 model finetuned on multilingual tasks. You'll learn how to initialize, modify, and train the model, gaining hands-on experience in efficient fine-tuning.

## Load the model and add PEFT adapter
First, we load the pretrained model by supplying the model name to the model loader `AutoModelForSeq2SeqLM`. Then add a PEFT adapter to the model using `get_peft_model`, which allows the model to maintain much of its pre-trained parameters while efficiently adapting to new tasks with a focused set of trainable parameters.


```python
from mindnlp.transformers import AutoModelForSeq2SeqLM
from mindnlp.peft import LoraConfig, TaskType, get_peft_model

# Load the pre-trained model
model_name_or_path = "bigscience/mt0-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

# Get the model with a PEFT adapter
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)

# Print the trainable parameters of the model
model.print_trainable_parameters()
```

`LoraConfig` specifies how the PEFT adapters should be configured:

* `task_type`: Defines the type of task, which in this case is TaskType.SEQ_2_SEQ_LM for sequence-to-sequence language modeling.
* `inference_mode`: A boolean that should be set to False when training to enable the training-specific features of the adapters.
* `r`: Represents the rank of the low-rank matrices that are part of the adapter. A lower rank means less complexity and fewer parameters to train.
* `lora_alpha`: LoRA alpha is the scaling factor for the weight matrices. A higher alpha value assigns more weight to the LoRA activations.
* `lora_dropout`: Sets the dropout rate within the adapter layers to prevent overfitting.

## Prepare the Dataset

To fine-tune the model, let's use the [financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) dataset. The financial_phrasebank dataset is specifically designed for sentiment analysis tasks within the financial sector. It contains sentences extracted from financial news articles, which are categorized based on the sentiment expressed — negative, neutral or positive.

Although the dataset is designed for sentiment classification task, we use it here for a sequence-to-sequence task for simplicity.

### Load the dataset
Load the dataset with `load_dataset` from MindNLP.

The data is then shuffled and split, allocating 90% for training and 10% for validation.


```python
from mindnlp.dataset import load_dataset

dataset = load_dataset("financial_phrasebank", "sentences_allagree")
train_dataset, validation_dataset = dataset.shuffle(64).split([0.9, 0.1])
```

### Add text label
Since we are training a sequence-to-sequence model, the output of the model needs to be text, which in our case is "negative", "neutral" or "positive". Therefore, we need to add a text label to in addition to the numeric label (0, 1 or 2) in each entry. This is achieved through the `add_text_label` function. The function is mapped onto each entry in the training and validation datasets through the `map` API.


```python
classes = dataset.source.ds.features["label"].names
def add_text_label(sentence, label):
    return sentence, label, classes[label.item()]

train_dataset = train_dataset.map(add_text_label, ['sentence', 'label'], ['sentence', 'label', 'text_label'])
validation_dataset = validation_dataset.map(add_text_label, ['sentence', 'label'], ['sentence', 'label', 'text_label'])
```

### Tokenization
We then tokenize the text with the tokenizer associated with the mT0 model. First, load the tokenizer:


```python
from mindnlp.transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
```

Next, modify the `BaseMapFunction` from MindNLP to wrap up the tokenization steps.

Note that both the `sentence` and the `text_label` columns needs to be tokenized.

In addition, to avoid unexpected behavior due to multiple threads attempting to tokenize data at the same time, we use `Lock` from the `threading` module to ensure only one thread can perform the tokenization at a time.


```python

import numpy as np
from mindnlp.dataset import BaseMapFunction
from threading import Lock
lock = Lock()

max_length = 128
class MapFunc(BaseMapFunction):
    def __call__(self, sentence, label, text_label):
        lock.acquire()
        model_inputs = tokenizer(sentence, max_length=max_length, padding="max_length", truncation=True)
        labels = tokenizer(text_label, max_length=3, padding="max_length", truncation=True)
        lock.release()
        labels = labels['input_ids']
        labels = np.where(np.equal(labels, tokenizer.pad_token_id), -100, labels)
        return model_inputs['input_ids'], model_inputs['attention_mask'], labels
```

Next, we apply the map function, shuffle the dataset if necessary and batch the dataset:


```python

def get_dataset(dataset, tokenizer, batch_size=None, shuffle=True):
    input_colums=['sentence', 'label', 'text_label']
    output_columns=['input_ids', 'attention_mask', 'labels']
    dataset = dataset.map(MapFunc(input_colums, output_columns),
                          input_colums, output_columns)
    if shuffle:
        dataset = dataset.shuffle(64)
    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset

batch_size = 8
train_dataset = get_dataset(train_dataset, tokenizer, batch_size=batch_size)
eval_dataset = get_dataset(validation_dataset, tokenizer, batch_size=batch_size, shuffle=False)
```

## Train the model

Now we have the model and datasets ready, let's prepare for the training.

### Optimizer and learning rate scheduler

We set up the optimizer for updating the model parameters, alongside a learning rate scheduler that manages the learning rate throughout the training process.


```python
from mindnlp.modules.optimization import get_linear_schedule_with_warmup
import mindspore.experimental.optim as optim

# Setting up optimizer and learning rate scheduler
optimizer = optim.AdamW(model.trainable_params(), lr=1e-3)

num_epochs = 3 # Number of iterations over the entire training dataset
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=(len(train_dataset) * num_epochs))
```

### Training step

Next, define the function that controls each step of training.

Define `forward_fn` which executes the model's forward pass to compute the loss.

Then pass `forward_fn` to `mindspore.value_and_grad` to create `grad_fn` that computes both the loss and gradients needed for parameter updates.

Define `train_step` that updates the model's parameters according to the computed gradients, which will be called in each step of training.


```python
import mindspore
from mindspore import ops

# Forward function to compute the loss
def forward_fn(**batch):
    outputs = model(**batch)
    loss = outputs.loss
    return loss

# Gradient function to compute gradients for optimization
grad_fn = mindspore.value_and_grad(forward_fn, None, model.trainable_params())

# Define the training step function
def train_step(**batch):
    loss, grads = grad_fn(**batch)
    optimizer(grads)  # Apply gradients to optimizer for updating model parameters
    return loss
```

### Training loop

Now everything is ready, let's implement the training and evaluation loop and excute the training process.

This process optimizes the model's parameters through multiple iterations over the dataset, i.e. multiple epochs, and assesses its performance on the evaluation dataset.


```python
from tqdm import tqdm

# Training loop across epochs
for epoch in range(num_epochs):
    model.set_train(True)
    total_loss = 0
    train_total_size = train_dataset.get_dataset_size()
    # Iterate over each entry in the training dataset
    for step, batch in enumerate(tqdm(train_dataset.create_dict_iterator(), total=train_total_size)):
        loss = train_step(**batch)
        total_loss += loss.float()  # Accumulate loss for monitoring
        lr_scheduler.step()  # Update learning rate based on scheduler

    model.set_train(False)
    eval_loss = 0
    eval_preds = []
    eval_total_size = eval_dataset.get_dataset_size()
    # Iterate over each entry in the evaluation dataset
    for step, batch in enumerate(tqdm(eval_dataset.create_dict_iterator(), total=eval_total_size)):
        with mindspore._no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.float()
        eval_preds.extend(
            tokenizer.batch_decode(ops.argmax(outputs.logits, -1).asnumpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataset)
    eval_ppl = ops.exp(eval_epoch_loss) # Perplexity
    train_epoch_loss = total_loss / len(train_dataset)
    train_ppl = ops.exp(train_epoch_loss) # Perplexity
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
```

Let's break down the training loop implementation and understand the key components:

* Model Training Mode

    Before the training starts, the model is set to training mode by `model.set_train(True)`. Before the evaluation, the training-specific behaviour of the model is disabled by `model.set_train(False)`.

* Loss and Perplexity

    `total_loss = 0` initializes and `total_loss += loss.float()` accumulates the total loss for each batch within an epoch. This accumulation is crucial for monitoring the model’s performance.

    The average loss and the perplexity (PPL), which is a common metric for language models, are reported in the printed message.

* Learning Rate Scheduler

    `lr_scheduler.step()` adjusts the learning rate after processing each batch, according to the predefined schedule. This is vital for effective learning, helping to converge faster or escape local minima.

* Evaluation Loop

    During evaluation, in addition to `model.set_train(False)`, `mindspore._no_grad()` ensures that gradients are not computed during the evaluation phase, which conserves memory and computations.
    The `tokenizer.batch_decode()` function converts the output logits from the model back into readable text. This is useful for inspecting what the model predicts and for further qualitative analysis.

## After training
Now that we have finished the training, we can assess its performance and save the trained model for future use.

### Accuracy compuation and check the predicited results

Let's comupute the accuracy of the predictions made on the validation dataset. Accuracy is a direct measure of how often the model's predictions match the actual labels, providing a straightforward metric to reflect the model's effectiveness.


```python
# Initialize counters for correct predictions and total predictions
correct = 0
total = 0

# List to store actual labels for comparison
ground_truth = []

# Compare each predicted label with the true label
for pred, data in zip(eval_preds, validation_dataset.create_dict_iterator(output_numpy=True)):
    true = str(data['text_label'])
    ground_truth.append(true)
    if pred.strip() == true.strip():
        correct += 1
    total += 1

# Calculate the percentage of correct predictions
accuracy = correct / total * 100

# Output the accuracy and sample predictions for review
print(f"{accuracy=} % on the evaluation dataset")
print(f"{eval_preds[:10]=}")
print(f"{ground_truth[:10]=}")
```

### Saving the model
If you are satisfied with the result, you can save the model as follows:


```python
# Save the model
peft_model_id = f"../../output/{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
model.save_pretrained(peft_model_id)
```

## Use the model for inference

Now let's load the saved model and demonstrate how to use it for making predictions on new data.

To load the model that has been trained with PEFT, we first load the base model with `AutoModelForSeq2SeqLM.from_pretrained`. On top of it, we add the trained PEFT adapter to the model with `PeftModel.from_pretrained`:


```python
from mindnlp.transformers import AutoModelForSeq2SeqLM
from mindnlp.peft import PeftModel, PeftConfig

peft_model_id = f"../../output/{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

# Load the model configuration
config = PeftConfig.from_pretrained(peft_model_id)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)

# Load the pretrained adapter
model = PeftModel.from_pretrained(model, peft_model_id)
```

Next, retrieve an entry from the validation dataset, or alternatively create an entry on your own.

We tokenize the `'sentence'` in this entry and use it as inputs into the model. Execute it and be curious about what the model will predict.


```python
# Retrieve an entry from the validation dataset.
# example = next(validation_dataset.create_dict_iterator(output_numpy=True)) # Get an example entry from the validation dataset
# print(example['sentence'])
# print(example['text_label'])

# Alternatively, create your own text
example = {'sentence': 'Nvidia Tops $3 Trillion in Market Value, Leapfrogging Apple.'}

inputs = tokenizer(example['sentence'], return_tensors="ms") # Get the tokenized text label
print(inputs)

model.set_train(False)
with mindspore._no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10) # Predict the text label using the trained model
    print(outputs)
    print(tokenizer.batch_decode(outputs.asnumpy(), skip_special_tokens=True)) # Print decoded text label from the prediction
```
