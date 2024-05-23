# Use Trainer

In the [Quick Start](./quick_start.md) tutorial, we have learned using the `Traienr` API to fine-tune a model.

This tutorial will give you a comprehensive description of configure `Trainer` to train your model for optimal outcomes.

The `TrainingArguments` and `Trainer` classes from MindNLP streamline the process of training machine learning models. `TrainingArguments` allows you to easily configure essential training parameters. `Trainer` then leverages these configurations to efficiently handle the entire training loop. Together, these tools abstract away much of the complexity of the training task, enabling both novices and experts to effectively optimize their models.

## Configure training parameters
By creating a `TrainingArguments` object, you can specify the desired configuration for the training process.

The following is a code snippet that instantiate a `TrainingArugments` object:


```python
from mindnlp.engine import TrainingArguments

training_args = TrainingArguments(
    output_dir="../../output",
    num_train_epochs=3,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50
)

```

Let's break down the code to understand each argument in detail.
### Basic parameters
* `output_dir`

  This parameter specifies the directory where the model's checkpoints and training outputs will be saved.
* `num_train_epochs`

  This parameter defines the total number of training cycles through the entire dataset that the model will undergo.

  Adjusting the number of training epochs directly impacts how well the model learns from the dataset. A higher number of epochs allows the model to learn more from the training data and achieve better results. However, setting too many epochs can lead to overfitting, where the model performs well on training data but poorly on new, unseen data.

### Optimizer parameters
`TrainingArguments` allows you to specify parameters of the optimizer, these include:

* `optim`

  This parameter specifies the optimizer to be used for training the model. So far, MindNLP supports AdamW and SGD. You can choose the optimizer by setting this parameter to `"adamw"` or `"sgd"`. By default, `TrainingArguments` chooses AdamW.

* `learning_rate`
  
  This patameter sets the initial learning rate for the optimizer, determining the step size at each iteration during loss minimization.

  This is one of first parameters to check if your training process fails to converge properly.
      
  A higher learning rate can converge faster, but if too high, it may overshoot the minimum, causing instability by bouncing around or diverging from the optimal weights. Conversely, a too-low learning rate can lead to slow convergence, potentially getting stuck in local minima.

* Advanced parameters for optimizer

  Using default values of the following advanced parameters suffices for most of the training. Interested readers and experts can adjust them to achieve better training result.


    - `weight_decay`: This parameter helps prevent overfitting by penalizing large weights. Weight decay is a regularization term added to the loss function, which effectively reduces the magnitude of weights in the model.

    - `adam_beta1` and `adam_beta2`: These parameters are specific to the AdamW optimizer. `adam_beta1` controls the exponential decay rate for the first moment estimates (similar to the momentum term), and `adam_beta2` controls the exponential decay rate for the second-moment estimates (related to the adaptive learning rate).

    - `adam_epsilon`: This is a very small number to prevent any division by zero in the implementation of the Adam optimizer. It's used to improve numerical stability.

    - `max_grad_norm`: This is used for gradient clipping, a technique to prevent exploding gradients in deep neural networks. Clipping the gradients at a specified norm helps in stabilizing the training process.

### Batch size parameters
Parameters related to batch size allow you to control how many examples are processed at a time during training and evaluation phases. Here's a summary of these parameters:

* `per_device_train_batch_size`

  This parameter sets the batch size for each training step.

  A large batch size can speed up training and make updates more consistent, but it might need more memory and could potentially converge to suboptimal minima.

  On the other hand, a smaller batch size requires less memory and might help the model learn better, though it could slow down the training process.

* `per_device_eval_batch_size`

  This parameter sets the batch size for each step in the evaluation.

Note: if you already batched your dataset before hand, by calling `dataset.batch()` for example, you would want to set the batch size in the `TrainingArguments` to 1, so the `Trainer` will not further batch on top your already batched dataset.

### Strategies for evaluation, saving and logging

The `TrainingArguments` allows you to define the strategies for evaluation, saving and logging during the training process.

#### Evaluation strategy

The `evaluation_strategy` parameter determines when the model should be evaluated during the training process. Evalutation is essential for monitoring model performance on a validation dataset, which is normally different from the training dataset. 

The strategy of performing evaluation can be:
- "no": No evaluation is performed.
- "steps": Evaluation occurs at specified intervals in terms of training steps.
  If the "steps" strategy is chosen, `eval_steps` needs to specified to control how many steps of training should occur between each evaluation.
- "epoch": Evaluation happens at the end of each epoch.

#### Save strategy

The `save_strategy` parameter controls when the model's state should be saved during the training process. Saving is crucial for preserving model checkpoints at different stages of training, which can be useful for recovery or further fine-tuning.

The strategy for saving can be:
- "no": No saving is performed.
- "steps": Saving occurs at specified intervals in terms of training steps.
  If the "steps" strategy is chosen, `save_steps` needs to be specified to control how many steps of training should occur between each saved checkpoint.
- "epoch": Saving happens at the end of each epoch.


#### Logging strategy
The `logging_strategy` parameter determines when the model's training metrics should be logged during the training process. Logging is important for tracking progress, understanding model behavior, and diagnosing issues during training.

The strategy for logging can be:
- "no": No logging is performed.
- "steps": Logging occurs at specified intervals in terms of training steps.
  If the "steps" strategy is chosen, the logging_steps needs to be specified to control how many steps of training should occur between each logging event.
- "epoch": Logging happens at the end of each epoch.

## Create the trainer

The `Trainer` in MindNLP accepts configurations from a `TrainingArgument` object and handle the entire training loop.

Assume you have defined the `model`, `dataset_train`, `dataset_val` and a function `compute_metrics`, for example as in the [Quick Start](./quick_start.md) tutorial, a `Trainer` object can be created using the following code:

```python
from mindnlp.engine import Trainer

trainer = Trainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    compute_metrics=compute_metrics,
    args=training_args,
)
```

Here's an explanation of key arguments used to customize training behavior:

* `model`: Pass the model instance you plan to train. This is the primary object that will undergo the training process.
* `args`: Your `TrainingArgument` object that sets up configurations for training.
* `train_dataset`, `eval_dataset`: These are the datasets on which the model will be trained or evaluated, respectively. Rember to preprocess the dataset as in the [Data Preprocess](./data_preprocess.md) tutorial.

* `compute_metrics`: A function that calculates specific performance metrics from the model's predictions. It takes a `mindnlp.engine.utils.EvalPrediction` object, which contains the predictions and labels, and returns the metric results.

  An example of `compute_metrics` function can be defined as follows:
  ```python
  import evaluate
  import numpy as np
  from mindnlp.engine.utils import EvalPrediction

  metric = evaluate.load("accuracy")

  def compute_metrics(eval_pred: EvalPrediction):
      logits, labels = eval_pred
      predictions = np.argmax(logits, axis=-1)
      return metric.compute(predictions=predictions, references=labels)
  ```

  Note that at the moment we still need to load the accuracy metric fromthe Hugging Face evaluate module.

Once the trainer is created, run `trainer.train()` to start your training process.
