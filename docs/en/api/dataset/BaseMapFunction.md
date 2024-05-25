# MindNLP.dataset.BaseMapFunction

???+ note "Definition"
    Class BaseMapFunction(input_colums, output_columns)

This class is a basic mapping function that maps input data to output data by specifying input and output data columns. Its function is to process the input data and return the processed result. But the function is not implemented here, and it needs to be inherited to implement its \__call__ method.

Args:

- **Input_colums**(list[str]) : Columns of input data to be passed
- **Output_columns**(list[str]): Columns returned after calling the object





Example:

```python
import mindspore as ms
from mindnlp.dataset import BaseMapFunction

class ModifiedMapFunction(BaseMapFunction):
    def __call__(self, text, label):
        tokenized = tokenizer(text, max_length=512, padding='max_length', truncation=True)
        labels = label.astype(ms.int32)
        return tokenized['input_ids'], tokenized['token_type_ids'], tokenize['attention_mask'], labels

map_fn = ModifiedMapFunction(['text', 'label'], ['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
```

By modifying the BaseMapFunction class, we created our own map function (`ModifiedMapFunction`).

The modified map function will take the text and label from each entry, tokenize the text, cast the label into type `Int32` and output the input_ids, token_type_ids, attention_mask and labels.

Note that the names of input and output columns are defined only when the map function is instantiated.

Let's now pass the `map_fn` into the `Trainer` together with other arguments:

```python
from mindnlp.engine import Trainer, TrainingArguments
from mindnlp.transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)
training_args = TrainingArguments(
    output_dir='../../output',
    per_device_train_batch_size=16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=imdb_train,
    map_fn=map_fn,
)
```