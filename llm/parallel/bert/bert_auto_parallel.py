import argparse
import json
import copy

import mindspore
from mindspore import nn
from mindspore.nn import AdamWeightDecay
from mindspore.dataset import  NumpySlicesDataset, SequentialSampler

from mindnlp.models import BertTokenizer, BertConfig, MSBertForSequenceClassification

from mindspore.communication import init, get_rank, get_group_size
from mindspore.train import Model
from mindspore.train import LossMonitor
from mindnlp.dataset import load_dataset



class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label,input_len):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_dataset_to_examples(ds):
    """Convert dataset to examples."""
    examples = []
    iter = ds.create_tuple_iterator()
    for i, ( text_a, text_b, label, idx) in enumerate(iter):
        # print(str(text_a.asnumpy()), str(text_b.asnumpy()))
        examples.append(
            InputExample(guid=i, text_a=str(text_a.asnumpy()), text_b=str(text_b.asnumpy()), label=int(label))
        )

    return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, tokenizer, max_seq_length=512, num_labels=2):
    features = []

    for ex_index, example in enumerate(examples):
        tokenizer.return_token = True
        tokens_a = tokenizer(example.text_a)
        tokens_b  = None
        if example.text_b:
            tokens_b = tokenizer(example.text_b)
        if tokens_b is not None:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        token_type_ids = []
        for token in tokens_a:
            tokens.append(token)
            token_type_ids.append(0)

        if tokens_b is not None:
            for token in tokens_b[1:]:
                tokens.append(token)
                token_type_ids.append(1)
            # tokens.append("[SEP]")
            # token_type_ids.append(1)

        tokenizer.return_token=False
        # input_ids = tokenizer.execute_py(example.text_a).tolist() + tokenizer.execute_py(example.text_b).tolist()
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # print(tokenizer.execute_py(np.array(tokens)).tolist())
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)
        input_len = len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        label_id = [ 0 for _ in range(num_labels)]
        label_id[example.label] = 1
        #label_id = example.label

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label_id,
                          input_len=input_len)
        )
    return features


def load_examples(tokenizer, max_seq_length):
    """load_examples using load_dataset"""
    mrpc_train = load_dataset('glue', 'mrpc', split='train')
    mrpc_test = load_dataset('glue', 'mrpc', split='test')

    train_examples = convert_dataset_to_examples(mrpc_train)
    test_examples = convert_dataset_to_examples(mrpc_test)

    train_features = convert_examples_to_features(train_examples, tokenizer, max_seq_length=max_seq_length)
    test_features = convert_examples_to_features(test_examples, tokenizer, max_seq_length=max_seq_length)

    # Convert to Tensors and build dataset
    train_all_input_ids = [f.input_ids for f in train_features]
    train_all_attention_mask = [f.attention_mask for f in train_features]
    train_all_token_type_ids = [f.token_type_ids for f in train_features]
    train_all_lens = [f.input_len for f in train_features]
    train_all_labels = [f.label for f in train_features]

    train_dataset = ((train_all_input_ids, train_all_attention_mask, train_all_token_type_ids, train_all_lens, train_all_labels))

    test_all_input_ids = [f.input_ids for f in test_features]
    test_all_attention_mask = [f.attention_mask for f in test_features]
    test_all_token_type_ids = [f.token_type_ids for f in test_features]
    test_all_lens = [f.input_len for f in test_features]
    test_all_labels = [f.label for f in test_features]
    test_dataset = ((test_all_input_ids, test_all_attention_mask, test_all_token_type_ids, test_all_lens, test_all_labels))

    return train_dataset, test_dataset

def parallel_setup():
    init("nccl")
    device_num = get_group_size()
    rank = get_rank()
    print("rank_id is {}, device_num is {}".format(rank, device_num))
    mindspore.reset_auto_parallel_context()
    mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.AUTO_PARALLEL, search_mode="sharding_propagation")
    mindspore.set_context(mode=mindspore.GRAPH_MODE)
    mindspore.set_context(device_target="GPU")

if __name__ == "__main__":
    parallel_setup()

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str, help="bert-base-cased")
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--lr", default=1e-4, type=float, help="Set 2e-5 for full-finetuning.")
    parser.add_argument("--max_seq_len", default=256, type=int)

    args = parser.parse_args()

    task = "mrpc"

    MODEL_CLASSES = {
        "bert-base-cased": (BertConfig, MSBertForSequenceClassification, BertTokenizer),
    }

    config_class, model_class, token_class = MODEL_CLASSES[args.model_name_or_path]

    tokenizer = token_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)


    train_ds, _ = load_examples(tokenizer, args.max_seq_len)
    train_sampler = SequentialSampler()
    train_dataloader = NumpySlicesDataset(train_ds, sampler=train_sampler)
    train_dataloader = train_dataloader.batch(args.batch_size)

    class MyTrainNet(nn.Cell):
        def __init__(self, backbone, loss_fn):
            super(MyTrainNet, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn
        def construct(self, input_ids, attention_mask, token_type_ids, lens, label):
            output = self._backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
            label = label.to(mindspore.float32)
            logits = output[0]
            return self._loss_fn(logits, label)
        @property
        def backbone_network(self):
            return self._backbone

    optimizer = AdamWeightDecay(params=model.trainable_params(), learning_rate=args.lr)
    loss = nn.CrossEntropyLoss()
    model = MyTrainNet(model, loss)
    model = nn.TrainOneStepCell(model, optimizer=optimizer)
    model = Model(network=model, loss_fn=None, optimizer=None)

    model.build(train_dataloader, epoch=args.num_epochs)
    model.train(epoch=args.num_epochs, train_dataset=train_dataloader, callbacks=LossMonitor(per_print_times=1))
