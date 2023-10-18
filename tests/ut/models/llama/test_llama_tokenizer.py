"""Test the CodeGenTokenizer"""

import mindspore as ms
from mindspore.dataset import GeneratorDataset
from mindnlp.transformers import LlamaTokenizer

LLAMA_SUPPORT_LIST = [
    "meta-llama/Llama-2-7b-hf",
]

TEXT = 'i make a small mistake when i\'m working! 床前明月光'
tokens = ['▁i', '▁make', '▁a', '▁small', '▁mistake', '▁when', '▁i', "'", 'm', '▁working', '!', '▁', '<0xE5>', '<0xBA>', '<0x8A>', '前', '明', '月', '光']
token_ids = [474, 1207, 263, 2319, 10171, 746, 474, 29915, 29885, 1985, 29991, 29871, 232, 189, 141, 30658, 30592, 30534, 30867]

def test_tokenizer_from_pretrained():
    # NOTE: THIS must be failed due to the huggingface authentification issue.
    """test CodeGenTokenizer from pretrained."""
    test_dataset = GeneratorDataset([TEXT], 'text')

    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_SUPPORT_LIST[0], return_token=True)

    assert tokenizer(TEXT) == tokens

    test_dataset = test_dataset.map(operations=tokenizer)
    dataset_after = next(test_dataset.create_tuple_iterator())[0]
    assert len(dataset_after) == 19
    assert dataset_after.dtype == ms.string

def test_tokenizer_convert_tokens_to_ids():
    """test convert tokens to ids."""
    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_SUPPORT_LIST[0], return_token=True)
    output_ids = tokenizer.convert_tokens_to_ids(tokens)
    assert output_ids == token_ids

def test_convert_ids_to_tokens():
    """test convert ids to tokens."""
    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_SUPPORT_LIST[0], return_token=True)
    ouput_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    assert ouput_tokens == tokens

def test_convert_tokens_to_string():
    """test convert tokens to string."""
    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_SUPPORT_LIST[0], return_token=True)
    output_string = tokenizer.convert_tokens_to_string(tokens)
    assert output_string == TEXT

def test_batch_encode():
    """test batch encode."""
    batch_texts = [
        'i make a small mistake when',
        ' i\'m working!',
        ' 床前明月光'
    ]
    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_SUPPORT_LIST[0], return_token=True)

    batched_ids = tokenizer.batch_encode(
        batch_texts, max_length=100, return_pts=True
    )
    print(batched_ids)
    assert batched_ids['input_ids'].shape == (3, 100)
    assert batched_ids['attention_mask'].shape == (3, 100)
