'''Copyright 2024 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Run the KTO training script with the commands below. In general, the optimal configuration for KTO will be similar to that of DPO.
'''

from dataclasses import dataclass

from mindnlp.dataset import load_dataset
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer

from mindnlp.trl import KTOConfig, KTOTrainer, setup_chat_format


config = KTOConfig(output_dir='op',
overwrite_output_dir = True,
warmup_ratio=0.1,
lr_scheduler_type='cosine',
per_device_train_batch_size=8,
                        per_device_eval_batch_size=8,
                        num_train_epochs=1,
                        gradient_accumulation_steps=2,
                        learning_rate=2e-4,
                        remove_unused_columns=False,
                        optim="adafactor",
                        logging_steps=250,
                        eval_steps=250,)
# parser = HfArgumentParser((ScriptArguments, Ktoconfig, ModelConfig))
# script_args, kto_args, model_args = parser.parse_args_into_dataclasses()

config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

# load model and dataset - dataset needs to be in a specific format
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# If we are aligning a base model, we use ChatML as the default template
if tokenizer.chat_template is None:
    model, tokenizer = setup_chat_format(model, tokenizer)

# Load the dataset
dataset = load_dataset("trl-lib/kto-mix-14k")
# raw_datasets = load_dataset("Anthropic/hh-rlhf")

# Apply chat template
dataset["prompt"] = tokenizer.apply_chat_template(dataset["prompt"], tokenize=False)
dataset["completion"] = tokenizer.apply_chat_template(dataset["completion"], tokenize=False)


# Initialize the KTO trainer
kto_trainer = KTOTrainer(
        model,
        # ref_model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        # peft_config=get_peft_config(model_args),
)

# Train and push the model to the Hub
kto_trainer.train()

