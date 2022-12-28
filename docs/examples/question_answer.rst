Question & Answer
=================

`GitHub <https://github.com/mindspore-lab/mindnlp/blob/master/examples/question_answer.py>`__

This section introduces a Question-Answering(QA) task in machine reading
comprehension(MRC), also called answer extraction: Given a passage of
text and a question, the machine is required to find a continuous
segment from the text as the answer according to the question. The
following is a demo that uses the SQuAD dataset and the Bi-Directional
Attention Flow model to train the QA task as an example:

.. note::

    This tutorial recommends using a GPU for experiments.

**SQuAD Dataset**

The SQuAD data set is very famous. It is a data set launched by Stanford
University in 2016, which is a reading comprehension data set. Given an
article, prepare the corresponding questions and need the algorithm to
give the answer to the question. All articles in this dataset are from
Wikipedia.

Here is a example item in training set:

+--------------+-----------------------------------------------------------+
|    column    | data                                                      |
+==============+===========================================================+
|    id        | 5733be284776f41900661182                                  |
+--------------+-----------------------------------------------------------+
|   context    | Architecturally, the school has a Catholic character.     |
|              | Atop the Main Building's gold dome is a golden statue of  |
|              | the Virgin Mary. Immediately in front of the Main         |
|              | Building and facing it, is a copper statue of Christ with |
|              | arms upraised with the legend "Venite Ad Me Omnes". Next  |
|              | to the Main Building is the Basilica of the Sacred Heart. |
|              | Immediately behind the basilica is the Grotto, a Marian   |
|              | place of prayer and reflection. It is a replica of the    |
|              | grotto at Lourdes, France where the Virgin Mary reputedly |
|              | appeared to Saint Bernadette Soubirous in 1858. At the    |
|              | end of the main drive (and in a direct line that connects |
|              | through 3 statues and the Gold Dome), is a simple, modern |
|              | stone statue of Mary.                                     |
+--------------+-----------------------------------------------------------+
|   question   | To whom did the Virgin Mary allegedly appear in 1858 in   |
|              | Lourdes France?                                           |
+--------------+-----------------------------------------------------------+
|   answer     | Saint Bernadette Soubirous                                |
+--------------+-----------------------------------------------------------+
| answer_start | 515                                                       |
|              |                                                           |
+--------------+-----------------------------------------------------------+

This in a raw data in training set. The answer_start indicates the
beginning char position of the answer in the context. After the data
processing, two columns, s_idx and e_idx will be added as label columns
which indicate the beginning and ending word position of the answer. The
following is a demo that uses the SQuAD dataset and BiDAF model to train
the QA task as an example. When given context and question, s_idx and
e_idx will be predicted.

Procedure of this task
----------------------

Load Dataset
~~~~~~~~~~~~~~~

MindNLP provides APIs to load and process various common datasets such
as SQuAD, IMDB, Multi30K, AG_News, etc.

Call the function ``load()`` from ``dataset`` to load the SQuAD dataset.
Then the training set and development set of the SQuAD dataset will be
returned.

The code of loading dataset:

.. code:: python

   from mindnlp.dataset import load
   squad_train, squad_dev = load('squad1')

Process Data
~~~~~~~~~~~~~~~

First obtain the embeddings and the vocabulary of words, by calling the
function ``from_pretrained()`` from ``Glove``. And since there is no
ready_made vocabulary of chars, you can define one by yourself:

.. code:: python

   from mindnlp.modules import Glove

   word_embeddings, word_vocab = Glove.from_pretrained('6B', 100, special_tokens=["<unk>", "<pad>"])
   char_dic = {"<unk>": 0, "<pad>": 1, "e": 2, "t": 3, "a": 4, "i": 5, "n": 6,\
                       "o": 7, "s": 8, "r": 9, "h": 10, "l": 11, "d": 12, "c": 13, "u": 14,\
                       "m": 15, "f": 16, "p": 17, "g": 18, "w": 19, "y": 20, "b": 21, ",": 22,\
                       "v": 23, ".": 24, "k": 25, "1": 26, "0": 27, "x": 28, "2": 29, "\"": 30, \
                       "-": 31, "j": 32, "9": 33, "'": 34, ")": 35, "(": 36, "?": 37, "z": 38,\
                       "5": 39, "8": 40, "q": 41, "3": 42, "4": 43, "7": 44, "6": 45, ";": 46,\
                       ":": 47, "\u2013": 48, "%": 49, "/": 50, "]": 51, "[": 52}
   char_vocab = text.Vocab.from_dict(char_dic)

Then initialize the tokenizer:

.. code:: python

   from mindnlp.dataset.transforms import BasicTokenizer

   tokenizer = BasicTokenizer(True)

Next, we apply the function ``process()`` to get the processed training
set:

.. code:: python

   from mindnlp.dataset import process
   squad_train = process('squad1', squad_train, char_vocab, word_vocab, tokenizer=tokenizer,\
                      max_context_len=768, max_question_len=64, max_char_len=48,\
                      batch_size=64, drop_remainder=False )

Define Model
~~~~~~~~~~~~~~~

The code of defining the Bi-Directional Attention Flow(BiDAF) model by
using MindNLP:

.. code:: python

   import mindspore.nn as nn
   from mindspore import Tensor
   from mindspore import Parameter
   from mindspore.common.initializer import Uniform, HeUniform, initializer

   from mindnlp.abc import Seq2vecModel
   from mindnlp.modules.embeddings import Word2vec

   class Encoder(nn.Cell):
       """
       Encoder for BiDAF model
       """
       def __init__(self, char_vocab_size, char_vocab, char_dim, char_channel_size, char_channel_width, word_vocab,
                     word_embeddings, hidden_size, dropout):
           super().__init__()
           self.char_vocab = char_vocab
           self.char_dim = char_dim
           self.char_channel_width = char_channel_width
           self.char_channel_size = char_channel_size
           self.word_vocab = word_vocab
           self.hidden_size = hidden_size
           self.dropout = nn.Dropout(1 - dropout)
           self.init_embed = initializer(Uniform(0.001), [char_vocab_size, char_dim])
           self.embed = Parameter(self.init_embed, name='embed')

           # 1. Character Embedding Layer
           self.char_emb = Word2vec(char_vocab, init_embed=self.embed, dropout=0.0)
           self.char_conv = nn.SequentialCell(
               nn.Conv2d(1, char_channel_size, (char_dim, char_channel_width), pad_mode="pad",
                         weight_init=HeUniform(math.sqrt(5)), bias_init=Uniform(1 / math.sqrt(1))),
               nn.ReLU()
               )

           # 2. Word Embedding Layer
           self.word_emb = words_embeddings

           # highway network
           self.highway_linear0 = nn.Dense(hidden_size * 2, hidden_size * 2,
                                           weight_init=HeUniform(math.sqrt(5)),
                                           bias_init=Uniform(1 / math.sqrt(hidden_size * 2)),
                                           activation=nn.ReLU())
           self.highway_linear1 = nn.Dense(hidden_size * 2, hidden_size * 2,
                                           weight_init=HeUniform(math.sqrt(5)),
                                           bias_init=Uniform(1 / math.sqrt(hidden_size * 2)),
                                           activation=nn.ReLU())
           self.highway_gate0 = nn.Dense(hidden_size * 2, hidden_size * 2,
                                         weight_init=HeUniform(math.sqrt(5)),
                                         bias_init=Uniform(1 / math.sqrt(hidden_size * 2)),
                                         activation=nn.Sigmoid())
           self.highway_gate1 = nn.Dense(hidden_size * 2, hidden_size * 2,
                                         weight_init=HeUniform(math.sqrt(5)),
                                         bias_init=Uniform(1 / math.sqrt(hidden_size * 2)),
                                         activation=nn.Sigmoid())

           # 3. Contextual Embedding Layer
           self.context_LSTM = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size,
                                       bidirectional=True, batch_first=True, dropout=dropout)

       def construct(self, c_char, q_char, c_word, q_word, c_lens, q_lens):
           # 1. Character Embedding Layer
           c_char = self.char_emb_layer(c_char)
           q_char = self.char_emb_layer(q_char)

           # 2. Word Embedding Layer
           c_word = self.word_emb(c_word)
           q_word = self.word_emb(q_word)

           # Highway network
           c = self.highway_network(c_char, c_word)
           q = self.highway_network(q_char, q_word)

           # 3. Contextual Embedding Layer
           c, _ = self.context_LSTM(c, seq_length=c_lens)
           q, _ = self.context_LSTM(q, seq_length=q_lens)

           return c, q

       def char_emb_layer(self, x):
           """
           param x: (batch, seq_len, word_len)
           return: (batch, seq_len, char_channel_size)
           """
           batch_size = x.shape[0]
           # x: [batch, seq_len, word_len, char_dim]
           x = self.dropout(self.char_emb(x))
           # x: [batch, seq_len, char_dim, word_len]
           x = ops.transpose(x, (0, 1, 3, 2))
           # x: [batch * seq_len, 1, char_dim, word_len]
           x = x.view(-1, self.char_dim, x.shape[3]).expand_dims(1)
           # x: [batch * seq_len, char_channel_size, 1, conv_len] -> [batch * seq_len, char_channel_size, conv_len]
           x = self.char_conv(x).squeeze(2)
           # x: [batch * seq_len, char_channel_size]
           x = ops.max(x, axis=2)[1]
           # x: [batch, seq_len, char_channel_size]
           x = x.view(batch_size, -1, self.char_channel_size)

           return x

       def highway_network(self, x1, x2):
           """
           param x1: (batch, seq_len, char_channel_size)
           param x2: (batch, seq_len, word_dim)
           return: (batch, seq_len, hidden_size * 2)
           """
           # [batch, seq_len, char_channel_size + word_dim]
           x = ops.concat((x1, x2), axis=-1)
           h = self.highway_linear0(x)
           g = self.highway_gate0(x)
           x = g * h + (1 - g) * x
           h = self.highway_linear1(x)
           g = self.highway_gate1(x)
           x = g * h + (1 - g) * x

           # [batch, seq_len, hidden_size * 2]
           return x


   class Head(nn.Cell):
       """
       Head for BiDAF model
       """
       def __init__(self, hidden_size, dropout):
           super().__init__()
           # 4. Attention Flow Layer
           self.att_weight_c = nn.Dense(hidden_size * 2, 1,
                                        weight_init=HeUniform(math.sqrt(5)),
                                        bias_init=Uniform(1 / math.sqrt(hidden_size * 2)))
           self.att_weight_q = nn.Dense(hidden_size * 2, 1,
                                        weight_init=HeUniform(math.sqrt(5)),
                                        bias_init=Uniform(1 / math.sqrt(hidden_size * 2)))
           self.att_weight_cq = nn.Dense(hidden_size * 2, 1,
                                         weight_init=HeUniform(math.sqrt(5)),
                                         bias_init=Uniform(1 / math.sqrt(hidden_size * 2)))
           self.softmax = nn.Softmax(axis=-1)
           self.batch_matmul = ops.BatchMatMul()

           # 5. Modeling Layer
           self.modeling_LSTM1 = nn.LSTM(input_size=hidden_size * 8, hidden_size=hidden_size,
                                         bidirectional=True, batch_first=True, dropout=dropout)
           self.modeling_LSTM2 = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size,
                                         bidirectional=True, batch_first=True, dropout=dropout)

           # 6. Output Layer
           self.p1_weight_g = nn.Dense(hidden_size * 8, 1,
                                       weight_init=HeUniform(math.sqrt(5)),
                                       bias_init=Uniform(1 / math.sqrt(hidden_size * 8)))
           self.p1_weight_m = nn.Dense(hidden_size * 2, 1,
                                       weight_init=HeUniform(math.sqrt(5)),
                                       bias_init=Uniform(1 / math.sqrt(hidden_size * 2)))
           self.p2_weight_g = nn.Dense(hidden_size * 8, 1,
                                       weight_init=HeUniform(math.sqrt(5)),
                                       bias_init=Uniform(1 / math.sqrt(hidden_size * 8)))
           self.p2_weight_m = nn.Dense(hidden_size * 2, 1,
                                       weight_init=HeUniform(math.sqrt(5)),
                                       bias_init=Uniform(1 / math.sqrt(hidden_size * 2)))

           self.output_LSTM = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size,
                                      bidirectional=True, batch_first=True, dropout=dropout)

       def construct(self, c, q, c_lens):
           # 4. Attention Flow Layer
           g = self.att_flow_layer(c, q)  #c, q are generated from Contextual Embedding Layer in Encoder

           # 5. Modeling Layer
           m, _ = self.modeling_LSTM2(self.modeling_LSTM1(g, seq_length=c_lens)[0], seq_length=c_lens)

           # 6. Output Layer
           p1, p2 = self.output_layer(g, m, c_lens)

           # [batch, c_len], [batch, c_len]
           return p1, p2

       def att_flow_layer(self, c, q):
           """
           param c: (batch, c_len, hidden_size * 2)
           param q: (batch, q_len, hidden_size * 2)
           return: (batch, c_len, q_len)
           """
           c_len = c.shape[1]
           q_len = q.shape[1]

           cq = []
           for i in range(q_len):
               # qi: [batch, 1, hidden_size * 2]
               qi = q.gather(mindspore.Tensor(i), axis=1).expand_dims(1)
               # ci: [batch, c_len, 1] -> [batch, c_len]
               ci = self.att_weight_cq(c * qi).squeeze(2)
               cq.append(ci)
           # cq: [batch, c_len, q_len]
           cq = ops.stack(cq, -1)

           # s: [batch, c_len, q_len]
           s = self.att_weight_c(c).broadcast_to((-1, -1, q_len)) + \
               self.att_weight_q(q).transpose((0, 2, 1)).broadcast_to((-1, c_len, -1)) + cq

           # a: [batch, c_len, q_len]
           a = self.softmax(s)
           # c2q_att: [batch, c_len, hidden_size * 2]
           c2q_att = self.batch_matmul(a, q)
           # b: [batch, 1, c_len]
           b = self.softmax(ops.max(s, axis=2)[1]).expand_dims(1)
           # q2c_att: [batch, hidden_size * 2]
           q2c_att = self.batch_matmul(b, c).squeeze(1)
           # q2c_att: [batch, c_len, hidden_size * 2]
           q2c_att = q2c_att.expand_dims(1).broadcast_to((-1, c_len, -1))

           # x: [batch, c_len, hidden_size * 8]
           x = ops.concat([c, c2q_att, c * c2q_att, c * q2c_att], axis=-1)
           return x

       def output_layer(self, g, m, l):
           """
           param g: (batch, c_len, hidden_size * 8)
           param m: (batch, c_len ,hidden_size * 2)
           return: p1: (batch, c_len), p2: (batch, c_len)
           """
           # p1: [batch, c_len]
           p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze(2)
           # m2: [batch, c_len, hidden_size * 2]
           m2, _ = self.output_LSTM(m, seq_length=l)
           # p2: [batch, c_len]
           p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze(2)

           return p1, p2

   class BiDAF(Seq2vecModel):
       def __init__(self, encoder, head):
           super().__init__(encoder, head)
           self.encoder = encoder
           self.head = head

       def construct(self, c_char, q_char, c_word, q_word, c_lens, q_lens):
           c, q = self.encoder(c_char, q_char, c_word, q_word, c_lens, q_lens)
           p1, p2 = self.head(c, q, c_lens)
           return p1, p2

Instantiate Model
~~~~~~~~~~~~~~~~~~~~

First we should define some hyperparameters:

.. code:: python

   char_vocab_size = len(char_vocab.vocab())
   char_dim = 8
   char_channel_width = 5
   char_channel_size = 100
   hidden_size = 100
   dropout = 0.2
   lr = 0.5
   epoch = 6

Then instantiate model using the following code:

.. code:: python

   encoder = Encoder(char_vocab_size, char_vocab, char_dim, char_channel_size, char_channel_width, word_vocab,
                     word_embeddings, hidden_size, dropout)
   head = Head(hidden_size, dropout)
   net = BiDAF(encoder, head)

Define Loss and Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A loss function is needed when we train the model. We use
``CrossEntropyLoss`` provided by MindSpore to define a loss function:

.. code:: python

   class Loss(nn.Cell):
       def __init__(self):
           super().__init__()

       def construct(self, logit1, logit2, s_idx, e_idx):
           loss_fn = nn.CrossEntropyLoss()
           loss = loss_fn(logit1, s_idx) + loss_fn(logit2, e_idx)
           return loss

   loss = Loss()

Then define the optimizer:

::

   optimizer = nn.Adadelta(net.trainable_params(), learning_rate=lr)

Train Model
~~~~~~~~~~~~~~

After defining the network, the loss function, and the optimizer,
we employ :py:class:`~mindnlp.engine.trainer.Trainer`
to train the model.

.. code:: python

   from mindnlp.engine.trainer import Trainer

   trainer = Trainer(network=net, train_dataset=squad_train, epochs=epoch, loss_fn=loss, optimizer=optimizer)
   trainer.run(tgt_columns=["s_idx", "e_idx"], jit=True)
