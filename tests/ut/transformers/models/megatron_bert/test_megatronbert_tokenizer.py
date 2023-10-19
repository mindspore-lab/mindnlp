"""
test the MegatronBertTokenizer
"""
import unittest
import mindspore as ms
from mindspore.dataset import GeneratorDataset
from mindnlp.transformers import MegatronBertTokenizer


class TestMegatronBertTokenizer(unittest.TestCase):
    r"""
    Test MegatronBertTokenizer
    """

    def test_megatronbert_tokenizer(self):
        """test BertTokenizer from pretrained."""
        texts = ['i make a small mistake when i\'m working! 床前明月光']
        test_dataset = GeneratorDataset(texts, 'text')

        bert_tokenizer = MegatronBertTokenizer.from_pretrained('nvidia/megatron-bert-cased-345m', return_token=True)
        test_dataset = test_dataset.map(operations=bert_tokenizer)
        dataset_after = next(test_dataset.create_tuple_iterator())[0]

        assert len(dataset_after) == 18
        assert dataset_after.dtype == ms.string

    def test_bert_tokenizer_add_special_tokens(self):
        """test add special tokens."""
        bert_tokenizer = MegatronBertTokenizer.from_pretrained('nvidia/megatron-bert-cased-345m')
        cls_id = bert_tokenizer.token_to_id("[CLS]")

        assert cls_id is not None
