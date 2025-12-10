import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import random
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, context
from mindspore.dataset import GeneratorDataset
import pandas as pd
from mindhf.transformers import AutoTokenizer, AutoModelForSequenceClassification
from mindhf.engine import Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


# Data preprocessing function
def preprocess_function(examples, tokenizer, max_length=128):
    # Tokenize input text
    # Important: For BARThez, we need to ensure inputs are properly formatted
    texts = examples["review"]
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np"  # Return numpy arrays instead of lists
    )

    # Add labels to inputs dictionary
    if isinstance(examples["label"], (int, float)):
        inputs["labels"] = np.array([examples["label"]], dtype=np.int32)
    else:
        inputs["labels"] = np.array(examples["label"], dtype=np.int32)

    return inputs


# Function to compute evaluation metrics
def compute_metrics(eval_preds):
    predictions, labels = eval_preds

    # Process predictions
    if isinstance(predictions, tuple):
        # Use the first element (usually logits)
        predictions = predictions[0]

    try:
        pred_array = np.array(predictions)
        print(f"Prediction shape: {pred_array.shape}")

        if len(pred_array.shape) > 2:
            # For sequence classification, we typically only care about the logits of the first token
            predictions = pred_array[:, 0, :]
            print(f"Reshaped shape: {predictions.shape}")

        # Get indices of maximum probability class
        preds = np.argmax(predictions, axis=-1)
        print(f"Labels shape: {np.array(labels).shape}")
        print(f"Predicted classes: {preds[:10]}")  # Print first 10 predictions

    except Exception as e:
        print(f"Error processing predictions: {e}")
        # If conversion fails, use a zero array as fallback
        preds = np.zeros_like(labels)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    accuracy = accuracy_score(labels, preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# Load and process the Allocine dataset
def load_allocine_dataset(sample_ratio=0.1):
    """
    Load the Allocine dataset using Hugging Face's datasets library.
    The dataset contains "review" text and "label" labels (0 for negative, 1 for positive)

    Parameters:
        sample_ratio: Proportion of data to use, range (0,1]
    """
    # Load Allocine dataset
    dataset = load_dataset("allocine")

    # Take a subset of the dataset (10%)
    if sample_ratio < 1.0:
        train_subset = dataset["train"].shuffle(seed=42).select(range(int(len(dataset["train"]) * sample_ratio)))
        test_subset = dataset["test"].shuffle(seed=42).select(range(int(len(dataset["test"]) * sample_ratio)))

        dataset = {
            "train": train_subset,
            "test": test_subset
        }

    return dataset


# Create MindSpore dataset
def create_mindspore_dataset(dataset, tokenizer, batch_size=8):
    """
    Create a MindSpore dataset from a Hugging Face dataset

    Parameters:
        dataset: Hugging Face dataset
        tokenizer: Tokenizer for preprocessing
        batch_size: Batch size for training
    """
    # Process the entire dataset at once to get all features
    features = []

    for i in range(0, len(dataset), 100):  # Process in chunks to avoid memory issues
        batch = dataset[i:min(i + 100, len(dataset))]
        texts = batch["review"]
        labels = batch["label"]

        # Tokenize the texts
        encodings = tokenizer(
            texts,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )

        # Add each example to the features list
        for j in range(len(texts)):
            features.append({
                "input_ids": encodings["input_ids"][j],
                "attention_mask": encodings["attention_mask"][j],
                "labels": labels[j]
            })

    # Create a generator function
    def generator():
        for item in features:
            yield (
                Tensor(item["input_ids"], dtype=ms.int32),
                Tensor(item["attention_mask"], dtype=ms.int32),
                Tensor(item["labels"], dtype=ms.int32)
            )

    # Create and return the MindSpore dataset
    return GeneratorDataset(
        generator,
        column_names=["input_ids", "attention_mask", "labels"]
    ).batch(batch_size)


# Main program
def main():
    # Load model and tokenizer
    print("Loading BARThez model and tokenizer...")
    model_name = "moussaKam/barthez"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained("moussaKam/barthez-sentiment-classification",
                                                               num_labels=2)

    # Load data, use only 10% for quick experimentation
    print("Loading Allocine dataset (10%)...")
    dataset = load_allocine_dataset(sample_ratio=0.5)

    # Allocine dataset is already split into train and test sets
    train_dataset_raw = dataset["train"]
    test_dataset_raw = dataset["test"]

    print(f"Number of training samples: {len(train_dataset_raw)}")
    print(f"Number of test samples: {len(test_dataset_raw)}")

    # Data preprocessing and creating MindSpore datasets
    print("Preprocessing data and creating MindSpore datasets...")
    batch_size = 16  # Reduce batch size to decrease memory usage

    # Create MindSpore datasets directly from Hugging Face datasets
    train_dataset = create_mindspore_dataset(train_dataset_raw, tokenizer, batch_size=batch_size)
    val_dataset = create_mindspore_dataset(test_dataset_raw, tokenizer, batch_size=batch_size)

    # Define training parameters
    training_args = TrainingArguments(
        output_dir="./results_barthez_classification",
        evaluation_strategy="epoch",
        learning_rate=2e-6,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir="./logs",
        logging_strategy="epoch",
        logging_steps=10,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
    )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Start training
    print("Starting training...")
    try:
        trainer.train()

        # Save model
        output_dir = './barthez_allocine_mindspore_model/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Saving model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Model evaluation
        print("Performing final model evaluation...")
        eval_results = trainer.evaluate()
        print(f"Final evaluation results: {eval_results}")

        # Save training statistics (using trainer's state)
        if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
            stats_df = pd.DataFrame(trainer.state.log_history)
            stats_df.to_csv(os.path.join(output_dir, 'training_stats.csv'), index=False)
            print("Training statistics saved")

        print("\nTraining completed!")


    except Exception as e:
        print(f"Error during training: {e}")
        # Try to get detailed error information
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    ms.set_context(device_target="Ascend", device_id=2)
    main()