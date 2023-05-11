"""Test the GPT2Tokenizer"""

import mindspore as ms
from mindspore.dataset import GeneratorDataset
from mindnlp.transforms import GPT2Tokenizer


def test_gpt2_tokenizer_from_pretrained():
    """test GPTTokenizer from pretrained."""
    texts = ['i make a small mistake when i\'m working! 床前明月光']
    test_dataset = GeneratorDataset(texts, 'text')

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', return_token=True)
    test_dataset = test_dataset.map(operations=gpt2_tokenizer)
    dataset_after = next(test_dataset.create_tuple_iterator())[0]

    assert len(dataset_after) == 20
    assert dataset_after.dtype == ms.string

def test_gpt2_tokenizer_add_special_tokens():
    """test add special tokens."""
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    cls_id = gpt2_tokenizer.token_to_id("[CLS]")

    assert cls_id == gpt2_tokenizer.unk_token_id

    add_num = gpt2_tokenizer.add_special_tokens({
        'cls_token': "[CLS]"
    })

    assert add_num == 1

    cls_id = gpt2_tokenizer.token_to_id("[CLS]")
    assert cls_id == 50257
