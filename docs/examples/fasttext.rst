Fasttext
--------

In this paper, we will train a fasttext model that can complete the task
of text classification by using the pre-trained glove word vector.

Step
~~~~

There are several steps.

1.Define Model

2.Define Hyperparameters

3.Data Preprocessing

4.Instantiate Model

5.Training Process

(1)Define Model
~~~~~~~~~~~~~~~

.. code:: python

   import numpy as np
   from mindspore import nn, Tensor
   from mindspore.common import dtype as mstype
   from mindspore.common.initializer import XavierUniform
   from mindspore.dataset.text.utils import Vocab
   from mindnlp.modules.embeddings import Glove

   class FasttextModel(nn.Cell):
       """
       FastText model
       """

       def __init__(self, vocab_size, embedding_dims, num_class):
           super(FasttextModel, self).__init__()
           self.vocab_size = vocab_size
           self.embeding_dims = embedding_dims
           self.num_class = num_class

           self.embeding_func = Glove(vocab=Vocab.from_list(['default']),
                                      init_embed=Tensor(np.zeros([self.vocab_size, self.embeding_dims]), mstype.float32))

           self.fc = nn.Dense(self.embeding_dims, out_channels=self.num_class,
                              weight_init=XavierUniform(1)).to_float(mstype.float16)

       def construct(self, text):
           """
           construct network
           """

           src_token_length = len(text)
           text = self.embeding_func(text)

           embeding = text.sum(axis=1)

           embeding = Tensor.div(embeding, src_token_length)

           embeding = embeding.astype(mstype.float32)
           classifier = self.fc(embeding)
           classifier = classifier.astype(mstype.float32)

           return classifier

(2)Define Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~

The following are some of the required hyperparameters in the model
training process.

.. code:: python

   vocab_size = 1383812
   embedding_dims = 16
   num_class = 4
   lr = 0.001
   bucket_boundaries = [64, 128, 467]
   max_len = 467
   drop = 0.0

(3)Data Preprocessing
~~~~~~~~~~~~~~~~~~~~~

The agnews dataset will be used in this article and downloaded
automatically through the mindnlp API. In the preprocessing, the data is
cleaned and then sorted into buckets after lookup.

Load dataset:

.. code:: python

   from mindnlp.dataset import load

   ag_news_train, ag_news_test = load('ag_news', shuffle=True)

Initializes the vocab and tokenizer for preprocessing:

.. code:: python

   from mindnlp.modules import Glove
   from mindnlp.dataset.transforms import BasicTokenizer

   tokenizer = BasicTokenizer(True)
   embedding, vocab = Glove.from_pretrained('6B', 100)

The loaded dataset is preprocessed and divided into training and
validation:

.. code:: python

   from mindnlp.dataset import process

   ag_news_train = process('ag_news', ag_news_train, tokenizer=tokenizer, vocab=vocab, \
                           bucket_boundaries=bucket_boundaries, max_len=max_len, drop_remainder=True)
   ag_news_train, ag_news_valid = ag_news_train.split([0.7, 0.3])

(4)Instantiate Model
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # net
   net = FasttextModel(vocab_size, embedding_dims, num_class)

(5)Training Process
~~~~~~~~~~~~~~~~~~~

Set the loss, optimizer, metric.

.. code:: python

   loss = nn.NLLLoss(reduction='mean')
   optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)
   metric = Accuracy()

Get started with mindnlp’s built-in trainer.

.. code:: python

   from mindnlp.engine.trainer import Trainer

   # define trainer
   trainer = Trainer(network=net, train_dataset=ag_news_train, eval_dataset=ag_news_valid, metrics=metric,
                     epochs=5, loss_fn=loss, optimizer=optimizer)

   print("start train")
   trainer.run(tgt_columns="label", jit=False)
   # trainer.run()
   print("end train")
