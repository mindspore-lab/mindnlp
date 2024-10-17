'''
    This file is an example for sft method.
'''
# imports
from mindnlp.dataset import load_dataset
from mindnlp.engine import SFTTrainer
from mindnlp.transformers import GPTTokenizer
from mindnlp.transformers import GPTForSequenceClassification
# tokenizer
tokenizer = GPTTokenizer.from_pretrained('openai-gpt')
model = GPTForSequenceClassification.from_pretrained('openai-gpt', num_labels=2)

# get dataset
imdb_ds = load_dataset('imdb', split=['train', 'test'])
imdb_train = imdb_ds['train']
imdb_test = imdb_ds['test']

# get trainer
trainer = SFTTrainer(
    model = model,
    train_dataset=imdb_train,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,)

# train
trainer.train()
