'''
    This file is an example for reward trainning method.
'''
from mindnlp.dataset import load_dataset
from tqdm import tqdm
from mindnlp.transformers import GPTTokenizer
from mindnlp.transformers import GPTForSequenceClassification

from mindnlp.engine import RewardConfig, RewardTrainer
tqdm.pandas()

config = RewardConfig(output_dir='op',
                    overwrite_output_dir = True,
                    warmup_ratio=0.1,
                    lr_scheduler_type='cosine',
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=8,
                    num_train_epochs=1,
                    gradient_accumulation_steps=2,
                    learning_rate=2e-4,
                    remove_unused_columns=False,
                    logging_steps=250,
                    eval_steps=250,)
config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

################
# Model & Tokenizer
################
# tokenizer
tokenizer = GPTTokenizer.from_pretrained('openai-gpt')
model = GPTForSequenceClassification.from_pretrained('openai-gpt', num_labels=1)

################
# Dataset
################
raw_datasets = load_dataset("Anthropic/hh-rlhf")

def preprocess_function(example):
    '''
        preprocess dataset.
    '''
    tokenizer.add_special_tokens({'pad_token': '0'})
    tokenized_rejected = tokenizer(example,truncation=True, max_length=512,  padding='max_length')

    return tokenized_rejected['input_ids'], tokenized_rejected['attention_mask']

train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]

train_dataset = train_dataset.map(
    operations=[preprocess_function],
    input_columns="chosen",
    output_columns=["input_ids_chosen", "attention_mask_chosen"]
)

train_dataset = train_dataset.map(
    operations=[preprocess_function],
    input_columns="rejected",
    output_columns=["input_ids_rejected", "attention_mask_rejected"]
)
train_dataset = train_dataset.padded_batch(1, pad_info={
                'input_ids_chosen': (None, tokenizer.pad_token_id),
                'attention_mask_chosen': (None, 0),
                'input_ids_rejected': (None, tokenizer.pad_token_id),
                'attention_mask_rejected': (None, 0)})



################
# Training
################
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # peft_config=get_peft_config(model_config),
)
trainer.train()
