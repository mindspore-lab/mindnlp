import mindspore
from mindspore.dataset import GeneratorDataset, transforms
from mindspore import nn
from mindspore import ops
from mindnlp.transforms import ErnieTokenizer, PadTransform
from mindspore import context, Tensor


context.set_context(mode = context.PYNATIVE_MODE)

# prepare dataset
class SentimentDataset:
    """Sentiment Dataset"""

    def __init__(self, path):
        self.path = path
        self._labels, self._text_a = [], []
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            dataset = f.read()
        lines = dataset.split("\n")
        for line in lines[1:-1]:
            label, text_a = line.split("\t")
            self._labels.append(int(label))
            self._text_a.append(text_a)

    def __getitem__(self, index):
        return self._labels[index], self._text_a[index]

    def __len__(self):
        return len(self._labels)
    
def process_dataset(source, tokenizer, pad_value, max_seq_len=64, batch_size=32, shuffle=True):
    column_names = ["label", "text_a"]
    rename_columns = ["label", "input_ids"]
    
    dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle)
    # transforms
    pad_op = PadTransform(max_seq_len, pad_value=pad_value)
    type_cast_op = transforms.TypeCast(mindspore.int32)
    
    # map dataset
    dataset = dataset.map(operations=[tokenizer, pad_op], input_columns="text_a")
    dataset = dataset.map(operations=[type_cast_op], input_columns="label")
    # rename dataset
    dataset = dataset.rename(input_columns=column_names, output_columns=rename_columns)
    # batch dataset
    dataset = dataset.batch(batch_size)

    return dataset

tokenizer = ErnieTokenizer.from_pretrained("uie-base")
pad_value = tokenizer.token_to_id('[PAD]')

dataset_train = process_dataset(SentimentDataset("data/train.tsv"), tokenizer, pad_value)
dataset_val = process_dataset(SentimentDataset("data/dev.tsv"), tokenizer, pad_value)
dataset_test = process_dataset(SentimentDataset("data/test.tsv"), tokenizer, pad_value, shuffle=False)

from mindnlp.models import ErnieForSequenceClassification
from mindnlp._legacy.amp import auto_mixed_precision
from mindnlp.engine import Trainer, Evaluator
from mindnlp.engine.callbacks import CheckpointCallback, BestModelCallback
from mindnlp.metrics import Accuracy


model = ErnieForSequenceClassification.from_pretrained('/home/luul/.mindnlp/ernie-3.0-nano-zh', num_labels=3)
model = auto_mixed_precision(model, 'O1')
# for name, param in model.parameters_and_names():
#     print(f"Layer: {name} ")
#     print(f"size:{param}")

loss = nn.CrossEntropyLoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=2e-5)

metric = Accuracy()

# define callbacks to save checkpoints
ckpoint_cb = CheckpointCallback(save_path='checkpoint', ckpt_name='ernie_emotect', epochs=1, keep_checkpoint_max=2)
best_model_cb = BestModelCallback(save_path='checkpoint', ckpt_name='ernie_emotect_best', auto_load=True)

trainer = Trainer(network=model, train_dataset=dataset_train,
                  eval_dataset=dataset_val, metrics=metric,
                  epochs=5, loss_fn=loss, optimizer=optimizer, callbacks=[ckpoint_cb, best_model_cb],
                  jit=True)

trainer.run('label')

# evaluator = Evaluator(network=model, eval_dataset=dataset_test, metrics=metric)
# evaluator.run(tgt_columns="label")

# dataset_infer = SentimentDataset("data/infer.tsv")

# def predict(text, label=None):
#     label_map = {0: "消极", 1: "中性", 2: "积极"}

#     text_tokenized = Tensor([tokenizer.encode(text).ids])
#     logits = model(text_tokenized)
#     predict_label = logits[0].asnumpy().argmax()
#     info = f"inputs: '{text}', predict: '{label_map[predict_label]}'"
#     if label is not None:
#         info += f" , label: '{label_map[label]}'"
#     print(info)

# from mindspore import Tensor

# for label, text in dataset_infer:
#     predict(text, label)

# predict("家人们咱就是说一整个无语住了 绝绝子叠buff")