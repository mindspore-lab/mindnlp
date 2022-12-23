Sequence Tagging
=================

Sequence Tagging refers to the process of tagging each Token in the
sequence given an input sequence. Sequence tagging problems are usually
used for information extraction from text, including Word Segmentation,
Position Tagging, Named Entity Recognition, NER, etc. The following is a
demo that uses the CoNLL2000Chunking dataset and the Bi-LSTM+CRF model
to train the named entity recognition task as an example:

Define Model
------------

.. code:: python

   import math
   from mindspore import nn
   from mindspore.common.initializer import Uniform, HeUniform
   from mindnlp.abc import Seq2vecModel
   from mindnlp.modules import CRF

   class Head(nn.Cell):
       """ Head for BiLSTM-CRF model """
       def __init__(self, hidden_dim, num_tags):
           super().__init__()
           weight_init = HeUniform(math.sqrt(5))
           bias_init = Uniform(1 / math.sqrt(hidden_dim * 2))
           self.hidden2tag = nn.Dense(hidden_dim, num_tags,
                                      weight_init=weight_init, bias_init=bias_init)

       def construct(self, context):
           return self.hidden2tag(context)

   class BiLSTM_CRF(Seq2vecModel):
       """ BiLSTM-CRF model """
       def __init__(self, encoder, head, num_tags):
           super().__init__(encoder, head)
           self.encoder = encoder
           self.head = head
           self.crf = CRF(num_tags, batch_first=True)

       def construct(self, text, seq_length, label=None):
           output,_,_ = self.encoder(text)
           feats = self.head(output)
           res = self.crf(feats, label, seq_length)
           return res

Define Hyperparameters
----------------------

The following are some of the required hyperparameters in the model
training process.

.. code:: python

   embedding_dim = 16
   hidden_dim = 32

Data Preprocessing
------------------

The dataset was downloaded and preprocessed by calling the interface of
dataset in mindnlp.

Load datasets:

.. code:: python

   from mindnlp.dataset import CoNLL2000Chunking

   dataset_train,dataset_test = CoNLL2000Chunking()

Initializes the vocab for preprocessing:

.. code:: python

   from mindspore.dataset import text

   vocab = text.Vocab.from_dataset(dataset_train,columns=["words"],freq_range=None,top_k=None,
                                      special_tokens=["<pad>","<unk>"],special_first=True)

Process datasets:

.. code:: python

   from mindnlp.dataset import CoNLL2000Chunking_Process

   dataset_train = CoNLL2000Chunking_Process(dataset=dataset_train, vocab=vocab,
                                             batch_size=32, max_len=80)

Instantiate Model
-----------------

.. code:: python

   from mindnlp.modules import RNNEncoder

   embedding = nn.Embedding(vocab_size=len(vocab.vocab()), embedding_size=embedding_dim,
                            padding_idx=vocab.tokens_to_ids("<pad>"))
   lstm_layer = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
   encoder = RNNEncoder(embedding, lstm_layer)
   head = Head(hidden_dim, 23)
   net = BiLSTM_CRF(encoder, head, 23)

Define Optimizer
----------------

.. code:: python

   from mindspore import ops

   optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01, weight_decay=1e-4)
   grad_fn = ops.value_and_grad(net, None, optimizer.parameters)

Define Train Step
-----------------

.. code:: python

   def train_step(data, seq_length, label):
       """ train step """
       loss, grads = grad_fn(data, seq_length, label)
       loss = ops.depend(loss, optimizer(grads))
       return loss

Training Process
----------------

Now that we have completed all the preparations, we can begin to train
the model.

.. code:: python

   from tqdm import tqdm

   size = dataset_train.get_dataset_size()
   steps = size
   with tqdm(total=steps) as t:
       for batch, (data, seq_length, label) in enumerate(dataset_train.create_tuple_iterator()):
           loss = train_step(data, seq_length ,label)
           t.set_postfix(loss=loss)
           t.update(1)
