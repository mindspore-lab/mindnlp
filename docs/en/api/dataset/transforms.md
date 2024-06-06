# dataset.transforms

Dataset processing transforms.

## dataset.transforms.Truncate
???+ note "Definition"
	Class Truncate(max_seq_len)

Truncate the input sequence so that it does not exceed the maximum length.



Args:

- **max_seq_len** (int): Maximum allowable length.



Raises：

- **TypeError**: If `max_seq_len` is not of type int.
- **TypeError**: if `text_input` is not a text line in 1-D Numpy format
- **ValueError**:If max_seq_len value is less than or equal to 0.
- **RuntimeError** :If the data type of the input tensor is not bool, int, float, double, or str.



Examples:

```python
import mindspore.dataset as ds
from mindnlp.dataset.transforms import Truncate

dataset = ds.NumpySlicesDataset(data=[['a', 'b', 'c', 'd', 'e']], shuffle=False)
# Data before 
# ['a' 'b' 'c' 'd' 'e']
truncate = Truncate(max_seq_len=3)
dataset = dataset.map(operations=truncate)
# Data after
# ['a' 'b' 'c']
```





## dataset.transforms.AddToken

???+ note "Definition"
	Class AddToken(token, begin=True)

Add token to beginning or end of sequence. It is often used to add special tags to a text sequence to mark the beginning or end of the sequence when performing natural language processing tasks.



Args:

- **token** (str): The token to be added.
- **begin** (bool, optional): Choose the position where the token is inserted. If True, the token will be inserted at the beginning of the sequence. Otherwise, it willbe inserted at the end of the sequence. Default: ``True``.



Raises:

- **TypeError**: If `token` is not of type str.
- **TypeError**: If input not a text line in 1-D ndarray contains string
- **TypeError**: If `begin` is not of type bool.



Example：

```python
import mindspore.dataset as ds
from mindnlp.dataset.transforms import AddToken

dataset = ds.NumpySlicesDataset(data={"text": [['a', 'b', 'c', 'd', 'e']]})
# Data before
# ['a' 'b' 'c' 'd' 'e']
add_start_token_op = AddToken(token='<start>', begin = True)
dataset = dataset.map(operations=add_start_token_op)
# Data after
# ['<start>' 'a' 'b' 'c' 'd' 'e']
add_end_token_op = AddToken(token='<start>', begin = True)
dataset = dataset.map(operations=add_end_token_op)
# Data after
# ['<start>' 'a' 'b' 'c' 'd' 'e' '<end>']
```





## dataset.transforms.Lookup

???+ note "Definition"
	Class Lookup( *vocab*, *unk_token*, *return_dtype*=mstype.int32)

Look up a word into an id according to the input vocabulary table.



Args:

- **vocab** (Vocab): A vocabulary object.
- **unknown_token** (str, optional): Word is used for lookup. In case of the word is out of vocabulary (OOV), the result of lookup will be replaced with unknown_token. If the unknown_token is not specified or  it is OOV, runtime error will be thrown. Default: ``None``, means no unknown_token is specified.
- **return_dtype** (mindspore.dtype, optional): The data type that lookup operation maps string to. Default: mindspore.int32.



Raises:

- **TypeError**: If `vocab` is not of type text.Vocab.
- **TypeError**: If `unknown_token` is not of type string.
- **TypeError**: If `return_dtype` is not of type mindspore.dtype.



Example:

```python
import mindspore.dataset as ds
import mindspore.dataset.text as text
from mindnlp.dataset.transforms import Lookup

# Load vocabulary from list
vocab = text.Vocab.from_list(['a', 'b', 'c', 'd', 'e'])
# Use lookup operation to map token to ids
lookup_op = Lookup(vocab,None)
ids = lookup_op(["b", "c"])
print("lookup: ids",ids)

```



## dataset.transforms.BasicTokenizer

???+ note "Definition"
	Class BasicTokenizer(*lower_case*=False, *py_transform*=False)

Tokenize the input UTF-8 encoded string by specific rules.



Args:

- **lower_case** (bool, optional): Whether to perform lowercase processing on the text. If True, will fold the text to lower case and strip accented characters. If False, will only perform normalization on the text.Default: False.
- **py_transform** (bool, optional): Whether use python implementation. Default: `False`.



Raises:

- **TypeError**: If `lower_case` is not of type bool.
- **TypeError**: If `py_transform` is not of type bool.
- **TypeError**：If `text_input` is not a text line in 1-D numpy format
- **RuntimeError**: If dtype of input Tensor is not str.



Examlpe:

```python
from mindnlp.dataset.transforms import BasicTokenizer

tokenizer_op = BasicTokenizer()
text = "Welcom to China!"
tokenized_text = tokenizer_op(text)
print("tokenized_text:", tokenized_text)
# tokenized_text: ['Welcom', 'to', 'China', '!']
```



## dataset.transforms.PadTransform

???+ note "Definition"
	Class PadTransform(*max_length*: int, *pad_value*:int, *return_length*:bool = False)



Pad tensor to a fixed length with given padding value.



Args:

- **max_length** (int): Maximum length to pad to.
- **pad_value** (int): Value to pad the tensor with.
- **return_length** (bool): Whether return auxiliary sequence length.



Raises:

- **TypeError**: If `max_length` is not of type int.
- **TypeError**: If `pad_value` is not of type int.
- **TypeError**: If `return_length` is not of type bool.
- **TypeError**: If text_input is not a text line in 1-D ndarrya contains string.



Example:

```python
import numpy as np
from mindnlp.dataset.transforms import PadTransform

pad_transform = PadTransform(max_length=10, pad_value='\0', return_length=True)
text_input = np.array(['Hello', 'world', 'this', 'is', 'a', 'test'], dtype='object')
text_output, length = pad_transform(text_input)

print("The length of the text sequence before filling:")
print(length) 
# 6
print("Filled text sequence:")
print(text_output)
# ['Hello' 'world' 'this' 'is' 'a' 'test' '\x00' '\x00' '\x00' '\x00']
```



## dataset.transforms.JiebaTokenizer

???+ note "Definition"
	Class JiebaTokenizer(*dict_path*='', *custom_word_freq_dict*=None)



Split a Chinese sentence into words and return the position of the words.



Args:

- **dict_path**(str,optional):Used to specify a custom dictionary file path that the stutterer can load to expand its thesaurus to better identify specific fields or specific words.Defalut:`""`
- **custom_word_freq_dict**(bool,optional):Allows users to provide custom words and their frequency information, so that the stutterer can consider the weight of these words when segmenting, thus more accurately segmenting words.Default:`None`



Raises:

- **TypeError**: If `dict_path` is not of type str.
- **TypeError**: If `custom_word_freq_dict` is not of type bool.


???+ note "Methods"
    ### dataset.transforms.JiebaTokenizer.tokenize

    
    > def tokenize(*self*, *sentence*, *cut_all*=False, *HMM*=True)

    Args:

    - sentence(str) :Sentences that need to be partitioned
    - cut_all(bool,optional):When the cut_all is True, the full-mode word segmentation is used, and all possible words in the sentence are segmented, which may produce more word segmentation results, which are suitable for lexical division in some specific scenarios, but may produce redundant results.
    - HMM(bool,optional):HMM stands for Hidden Markov Model. When HMM is True, Stuttering Segmentation enables the HMM model to recognize new words, such as unlogged words. This can improve the accuracy of word segmentation, especially for some obscure or new words.

    Example:

    ```python
    from mindnlp.dataset.transforms import JiebaTokenizer

    tokenizer = JiebaTokenizer()
    sentence = "今天天气真好，适合出去玩。"
    tokens = tokenizer.tokenize(sentence)
    print(tokens)
    # ['今天天气', '真', '好', '，', '适合', '出去玩', '。']
    ```