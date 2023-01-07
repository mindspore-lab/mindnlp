Dataset and transforms
======================

Dataset
-------

In mindnlp, there are download interfaces for some datasets, which can
be used to download datasets directly. Based on the following
classifications, the datasets currently included are:

-  Machine Translation

   -  ​\ ``IWSLT2016``\ ​
   -  ​\ ``IWSLT2017``\ ​
   -  ​\ ``Multi30k``\ ​​

-  Question Answer

   -  ​\ ``SQuAD1``\ ​
   -  ​\ ``SQuAD2``\ ​​

-  Sequence Tagging

   -  ​\ ``CoNLL2000Chunking``\ ​
   -  ​\ ``UDPOS``\ ​

-  Text Classification

   -  ​\ ``AG_NEWS``\ ​
   -  ​\ ``AmazonReviewFull``\ ​
   -  ​\ ``AmazonReviewPolarity``\ ​
   -  ​\ ``CoLA``\ ​
   -  ​\ ``DBpedia``\ ​
   -  ​\ ``IMDB``\ ​
   -  ​\ ``MNLI``\ ​
   -  ​\ ``MRPC``\ ​
   -  ​\ ``QNLI``\ ​
   -  ​\ ``QQP``\ ​
   -  ​\ ``RTE``\ ​
   -  ​\ ``SogouNews``\ ​
   -  ​\ ``SST2``\ ​
   -  ​\ ``STSB``\ ​
   -  ​\ ``WNLI``\ ​
   -  ​\ ``YahooAnswers``\ ​
   -  ​\ ``YelpReviewFull``\ ​
   -  ​\ ``YelpReviewPolarity``\ ​

-  Text Generation

   -  ​\ ``LCSTS``\ ​
   -  ​\ ``PennTreebank``\ ​
   -  ​\ ``WikiText2``\ ​
   -  ​\ ``WikiText103``\ ​

Dataset Loading
---------------

There are two ways to load a dataset. The first is to call the
corresponding interface, and the second is to call a unified interface.

Method 1: Load by corresponding interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The corresponding interface can be found under ``mindnlp.dataset`` .
Here, the ``Multi30k`` dataset is used as an example:

.. code:: python

   from mindnlp.dataset import Multi30k

Parameter list and returns can be known through the annotation, or the
corresponding docs on mindnlp
`website <https://mindnlp.cqu.ai/en/latest/api/dataset/machine_translation.html#module-mindnlp.dataset.machine_translation.multi30k>`__:

Parameters:

-  **root** (*str*) - Directory where the datasets are saved.
   Default:'~/.mindnlp'
-  **split** (*str|Tuple[str]*) - Split or splits to be returned.
   Default:('train', 'valid', 'test').
-  **language_pair** (*Tuple[str]*) - Tuple containing src and tgt
   language. Default: ('de', 'en').
-  **proxies** (*dict*) - a dict to identify proxies,for example:
   {“https”: “https://127.0.0.1:7890”}.

Returns:

-  **datasets_list** (list) -A list of loaded datasets. If only one type
   of dataset is specified,such as 'trian', this dataset is returned
   instead of a list of datasets.

For convenience, we use default parameters except for the first one:

.. code:: python

   multi30k_train, multi30k_valid, multi30k_test = Multi30k("./dataset")

Doubtlessly, if you just want to pick the train dataset, you only need
to modify the parameters below:

.. code:: python

   multi30k_train = Multi30k(root="./dataset", split='train')

Method 2: Load by unified interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Also, we can load dataset by a unified interface - ``load()`` . The
first parameter is a string to specify a dataset:

.. code:: python

   from mindnlp.dataset import load
   multi30k_train, multi30k_valid, multi30k_test = load('multi30k')

The other parameter can be added sequentially according to the
interface:

.. code:: python

   multi30k_train, multi30k_valid, multi30k_test = load('multi30k', root="./dataset")

Customizing Dataset
^^^^^^^^^^^^^^^^^^^

If you want to use customizd dataset, more information about customizing
dataset could be found on mindspore
`website <https://www.mindspore.cn/tutorials/zh-CN/r1.9/beginner/dataset.html#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86>`__.

Dataset Iteration
-----------------

There are usually multiple columns in a dataset, and you can query the
column names using the ``get_col_names()`` interface:

.. code:: python

   dataset_train.get_col_names()

::

   ['de', 'en']

After the dataset is loaded, the data is obtained iteratively and then
sent to the neural network for training. We can use
``create_tuple_iterator()`` or ``create_dict_iterator()`` ​
interface to create an iterater for data access. Combining the column
names interface above:

.. code:: python

   for de_value, en_value in dataset_train.create_tuple_iterator():
       print(de_value)
       print(en_value)
       break

::

   Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.
   Two young, White males are outside near many bushes.

Data Transforms
---------------

Common Operation
^^^^^^^^^^^^^^^^

The most important operation in data transformation processing is the
``map`` operation. ``map`` can add a data transform to a specified
column in a dataset, make it apply to each element of the column data,
and then return the dataset after transformation.
``BasicTokenizer()`` is used here for word segmentation of two
columns of the dataset, and ``from_dataset`` is used to generate the
vocab:

.. code:: python

   from mindnlp.dataset.transforms import BasicTokenizer

   tokenizer = BasicTokenizer(True)
   dataset_train= dataset_train.map([tokenizer], 'en')
   dataset_train= dataset_train.map([tokenizer], 'de')

   en_vocab = text.Vocab.from_dataset(dataset_train, 'en', special_tokens=['<pad>', '<unk>'], special_first= True)
   de_vocab = text.Vocab.from_dataset(dataset_train, 'de', special_tokens=['<pad>', '<unk>'], special_first= True)
   vocab = {'en':en_vocab, 'de':de_vocab}

Data Preprocessing in mindnlp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are different processes for different data sets in different
domains. In mindnlp, specific processing functions are provided to help
us process data quickly. Similarly, two ways can be used to process
data. Using the ``Multi30k`` dataset as an example:

Method 1: Process by corresponding interface
""""""""""""""""""""""""""""""""""""""""""""

The corresponding interface can be found under ``mindnlp.dataset`` ,
the name of which begins with dataset's name, the underline and
``Process`` following. The ``vocab`` in the code was generated
above:

.. code:: python

   from mindnlp.dataset import Multi30k_Process
   train_dataset = Multi30k_Process(train_dataset, vocab=vocab)

Parameter list and returns can be known through the annotation, or the
corresponding docs on mindnlp
`website <https://mindnlp.cqu.ai/en/latest/api/dataset/machine_translation.html#module-mindnlp.dataset.machine_translation.multi30k>`__:

Parameters:

-  **dataset** ( *GeneratorDataset* ) - Multi30k dataset.
-  **vocab** ( *Vocab* ) - vocabulary object, used to store the
   mapping of token and index.
-  **batch_size** ( *int* ) - The number of rows each batch is
   created with. Default: 64.
-  **max_len** ( *int* ) - The max length of the sentence. Default:
   500.
-  **drop_remainder**  ( *bool* ) - When the last batch of data
   contains a data entry smaller than batch_size, whether to discard the
   batch and not pass it to the next operation. Default: False.

Returns:

-  **dataset** (MapDataset) - dataset after transforms.

Method 2: Process by unified interface
""""""""""""""""""""""""""""""""""""""

.. code:: python

   from mindnlp.dataset import process
   dataset_train = process('Multi30k', dataset_train, vocab = vocab)

For complete code, please check out the the github
`repository <https://github.com/mindspore-lab/mindnlp/blob/master/examples/machine_translation.py>`__

Customizing Preprocess
""""""""""""""""""""""

If you want to preprocess dataset by yourself, please refer to more
operations on mindspore
`website <https://www.mindspore.cn/tutorials/zh-CN/r1.9/beginner/transforms.html>`__.


