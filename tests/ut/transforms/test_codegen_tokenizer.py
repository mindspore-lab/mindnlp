"""Test the CodeGenTokenizer"""

import mindspore as ms
from mindspore.dataset import GeneratorDataset
from mindnlp.transforms import CodeGenTokenizer


def test_codegen_tokenizer_from_pretrained():
    """test CodeGenTokenizer from pretrained."""
    texts = ['i make a small mistake when i\'m working! 床前明月光']
    test_dataset = GeneratorDataset(texts, 'text')

    codegen_tokenizer = CodeGenTokenizer.from_pretrained('Salesforce/codegen-350M-mono', return_token=True)
    test_dataset = test_dataset.map(operations=codegen_tokenizer)
    dataset_after = next(test_dataset.create_tuple_iterator())[0]

    assert len(dataset_after) == 20
    assert dataset_after.dtype == ms.string


def test_codegen_tokenizer_add_special_tokens():
    """test add special tokens."""
    codegen_tokenizer = CodeGenTokenizer.from_pretrained('Salesforce/codegen-350M-mono')
    cls_id = codegen_tokenizer.token_to_id("[CLS]")

    assert cls_id is None

    add_num = codegen_tokenizer.add_special_tokens({
        'cls_token': "[CLS]"
    })

    assert add_num == 1

    cls_id = codegen_tokenizer.token_to_id("[CLS]")
    assert cls_id == 50295
