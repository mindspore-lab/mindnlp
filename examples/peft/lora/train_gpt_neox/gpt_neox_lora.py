# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Fine-tuning of instructions.
Fine-tuning GPT NeoX with BELLE Chinese instructional data.
"""
import os
import argparse
import mindspore
from mindspore import ops
from mindspore.amp import StaticLossScaler
from mindnlp.dataset import load_dataset
from mindnlp.transformers.models.gpt_neox import (
    GPTNeoXForCausalLM,
)
from mindnlp.engine import TrainingArguments, Trainer
from mindnlp.transformers import AutoTokenizer
from mindnlp.peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

BOS_TOKEN = None
EOS_TOKEN = None

def add_result_token(instruction, inputs, output):
    """
    Preprocess the dataset.
    """
    # Retrieve specific value through item.
    instruction = instruction.item()
    output = output.item()
    inputs = inputs.item()
    input_text = "Human: " + instruction + inputs + "\n\nAssistant: "
    input_text = BOS_TOKEN + input_text if BOS_TOKEN is not None else input_text
    response = input_text + output + EOS_TOKEN
    return response

class CausalLMTrainer(Trainer):
    """
    Used for GPTNeoX CausalLM training.
    """
    def __init__(self, **kwargs):
        """
        Modified from Trainer.
        """
        self.loss_scaler = StaticLossScaler(scale_value=args.scale_value)

        super().__init__(**kwargs)

    def training_step(self, model, inputs):
        """
        Modified from Trainer.

        Perform a training step on a batch of inputs.
        """
        model.set_train()
        inputs = self._prepare_inputs(inputs)

        def forward(inputs):
            loss = self.compute_loss(model, inputs)
            # Loss scale
            loss = self.loss_scaler.scale(loss)
            return loss

        if getattr(self, 'grad_fn', None) is None or self.model_reload:
            self.grad_fn = mindspore.value_and_grad(forward, None, self.optimizer.parameters)

        loss, grads = self.grad_fn(inputs)
        # Try using Loss scale
        loss = self.loss_scaler.unscale(loss)
        grads = self.loss_scaler.unscale(grads)

        return loss / self.args.gradient_accumulation_steps, grads

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Modified from Trainer.

        How the loss is computed by Trainer. By default, all models return the
        loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            # unwrapped_model = self.accelerator.unwrap_model(model)
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            shift_logits = outputs["logits"][:, :-1, :]
            labels = inputs["labels"][:, 1:]
            loss = ops.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]).to(mindspore.float32),
                labels.view(-1)
            ).to(mindspore.float16)

        return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':
    # Replace the HuggingFace download link with hf-mirror.
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-410m-deduped")

    parser = argparse.ArgumentParser(description="Train GPT-NeoX with LoRA")

    parser.add_argument("--max_length", type=int, default=512,
                        help="The maximum length of the sequence")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="The LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="The LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="The dropout rate for LoRA")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="The learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="The weight decay")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="The number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="The batch size")

    parser.add_argument("--logging_steps", type=int, default=100,
                        help="The number of steps between logging progress")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="The number of steps between saving the model")
    parser.add_argument("--save_total_limit", type=int, default=4,
                        help="The total number of checkpoints to keep")

    parser.add_argument("--scale_value", type=int, default=2 ** 5,
                        help="The scaling value for loss scale")
    parser.add_argument("--label_smoothing_factor", type=float, default=0.0,
                        help="The label smoothing factor")

    parser.add_argument("--output_dir", type=str, default="output",
                        help="The output directory")
    parser.add_argument("--resume_from_checkpoint", type=str,
                        default=None,
                        help="Path to resume checkpoint")

    args = parser.parse_args()

    # Load Belle dataset
    # https://huggingface.co/datasets/BelleGroup/train_0.5M_CN/blob/main/Belle_open_source_0.5M.json
    ds = load_dataset("BelleGroup/train_0.5M_CN")

    # Splitting the training set and test set.
    train_dataset, eval_dataset = ds.split([0.9, 0.1])

    # Set Prompt
    BOS_TOKEN = tokenizer.bos_token
    EOS_TOKEN = tokenizer.eos_token

    train_dataset = train_dataset.map(
        add_result_token,
        input_columns=['instruction', 'input', 'output'],
        output_columns=['inputs']
    )
    eval_dataset = eval_dataset.map(
        add_result_token,
        ['instruction', 'input', 'output'],
        ['inputs']
    )

    # set pad_token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def tokenize_prompt(inputs):
        """
        Convert the sentence into token ids using a tokenizer.
        """
        result = tokenizer(
            inputs.item(),
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result["input_ids"], result["attention_mask"], result["labels"]

    def dataset_batch(dataset, shuffle=False, buffer_size=16):
        """
        Split the dataset into batches and use dynamic padding to ensure
        that each batch has a consistent length.
        """
        if shuffle:
            dataset = dataset.shuffle(buffer_size).map(
                tokenize_prompt,
                ["inputs"],
                ["input_ids", "attention_mask", "labels"]
            )
        else:
            dataset = dataset.map(
                tokenize_prompt,
                ["inputs"],
                ["input_ids", "attention_mask", "labels"]
            )

        dataset = dataset.padded_batch(
            args.batch_size,
            pad_info={
                'input_ids': (None, tokenizer.pad_token_id),
                'attention_mask': (None, 0),
                'labels': (None, tokenizer.pad_token_id)
            }
        )
        return dataset

    train_dataset = dataset_batch(train_dataset)
    eval_dataset = dataset_batch(eval_dataset)

    # Creating peft model
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Set Training Param
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        label_smoothing_factor=args.label_smoothing_factor,
    )

    trainer = CausalLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
