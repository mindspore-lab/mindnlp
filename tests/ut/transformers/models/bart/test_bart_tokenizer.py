"""Test the BartTokenizer"""

import mindspore as ms
from mindspore.dataset import GeneratorDataset
from mindnlp.transformers import BartTokenizer


def test_bart_tokenizer_from_pretrained():
    """test BartTokenizer from pretrained."""
    texts = ['i make a small mistake when i\'m working! 床前明月光']
    test_dataset = GeneratorDataset(texts, 'text')

    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', return_token=True)
    test_dataset = test_dataset.map(operations=bart_tokenizer)
    dataset_after = next(test_dataset.create_tuple_iterator())[0]
    assert len(dataset_after) == 22
    assert dataset_after.dtype == ms.string

def test_bart_tokenizer_add_special_tokens():
    """test add special tokens."""
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    cls_id = bart_tokenizer.token_to_id("[CLS]")

    assert cls_id == bart_tokenizer.unk_token_id

    add_num = bart_tokenizer.add_special_tokens({
        'cls_token': "[CLS]"
    })

    assert add_num == 1

    cls_id = bart_tokenizer.token_to_id("[CLS]")
    assert cls_id == 50265
