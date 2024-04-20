from mindspore import ops

from mindnlp.transformers import BloomTokenizerFast, BloomForCausalLM
from mindnlp.engine import TrainingArguments, Trainer
from mindnlp.dataset import load_dataset, BaseMapFuction
from mindnlp.amp import autocast

class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=ops.ones_like(inputs["input_ids"]).bool(),
            labels=inputs["input_ids"],
        ).loss

model_name = "bloom-560m"
model = BloomForCausalLM.from_pretrained(f"bigscience/{model_name}")
tokenizer = BloomTokenizerFast.from_pretrained(f"bigscience/{model_name}", add_prefix_space=True)

dataset = load_dataset('tatsu-lab/alpaca')
print(dataset.get_col_names())

class ModifiedMapFunction(BaseMapFuction):
    def __call__(self, text):
        tokenized = tokenizer(text, max_length=512, padding="max_length", truncation=True)
        return tokenized['input_ids']

training_args = TrainingArguments(
    "output",
    fp16=False,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    num_train_epochs=2,
    logging_steps=100,
    save_strategy='epoch'
)

trainer = ModifiedTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    map_fn=ModifiedMapFunction('text', 'input_ids'),
)

trainer.train()
