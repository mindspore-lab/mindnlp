import json
import copy

from mindspore.dataset import NumpySlicesDataset, SequentialSampler
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
    def __init__(self, input_ids, attention_mask, token_type_ids, label, input_len):
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
    for (
        i,
        (text_a, text_b, label, _),
    ) in enumerate(iter):
        examples.append(
            InputExample(
                guid=i,
                text_a=str(text_a.asnumpy()),
                text_b=str(text_b.asnumpy()),
                label=int(label),
            )
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


def convert_examples_to_features(examples, tokenizer, max_seq_length=512):
    features = []

    for ex_index, example in enumerate(examples):
        tokenizer.return_token = True
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        if tokens_b is not None:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length)
        else:
            if len(tokens_a) > max_seq_length:
                tokens_a = tokens_a[0:max_seq_length]

        tokens = []
        token_type_ids = []

        for token in tokens_a:
            tokens.append(token)
            token_type_ids.append(0)

        if tokens_b is not None:
            for token in tokens_b:
                tokens.append(token)
                token_type_ids.append(1)

        tokenizer.return_token = False
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

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

        label_id = example.label

        if ex_index < 1:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            print("label: %s (id = %d)" % (example.label, label_id))
            print("input length: %d" % (input_len))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label_id,
                input_len=input_len,
            )
        )
    return features


def load_examples(tokenizer, max_seq_length, data_type="train"):
    """load_examples using load_dataset"""
    ds = load_dataset("glue", "mrpc")
    mrpc_train = ds["train"]
    mrpc_test = ds["test"]
    mrpc = mrpc_train if data_type == "train" else mrpc_test
    train_examples = convert_dataset_to_examples(mrpc)
    test_examples = convert_dataset_to_examples(mrpc_test)

    features = convert_examples_to_features(
        train_examples, tokenizer, max_seq_length=max_seq_length
    )

    # Convert to Tensors and build dataset
    all_input_ids = [f.input_ids for f in features]
    all_attention_mask = [f.attention_mask for f in features]
    all_token_type_ids = [f.token_type_ids for f in features]
    all_lens = [f.input_len for f in features]
    all_labels = [f.label for f in features]
    dataset = (
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_lens,
        all_labels,
    )

    return dataset


def get_dataloader_from_ds(ds, batch_size):
    train_sampler = SequentialSampler()
    train_dataloader = NumpySlicesDataset(ds, sampler=train_sampler)
    train_dataloader = train_dataloader.batch(batch_size)

    return train_dataloader
