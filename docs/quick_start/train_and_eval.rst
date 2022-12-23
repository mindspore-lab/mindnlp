Train and Eval
===================

We use the task of `sentiment analysis <https://mindnlpdocs-traineval.readthedocs.io/en/latest/examples/sentiment_analysis.html>`_
to give a detailed introduction. And we employ :py:class:`~mindnlp.engine.trainer.Trainer`
for a fast training and evaluation.

Load and Process Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Load Data
------------------------------------

We can call the function :py:class:`load()` from :py:class:`~mindnlp.dataset` to load
the IMDB dataset. And then the method will return the training set and the testing set
of the IMDB dataset.

The code of loading data is as follows:
.. code-block:: python
    from mindnlp.dataset import load

    imdb_train, imdb_test = load('imdb', shuffle=True)

Process and Split Data
------------------------------------

First we obtain the embeddings and the vocabulary, by calling the function
:py:class:`from_pretrained()` from
:py:class:`~mindnlp.modules.embeddings.glove_embedding.Glove`:
::
    from mindnlp.modules import Glove

    embedding, vocab = Glove.from_pretrained('6B', 100, special_tokens=["<unk>", "<pad>"], dropout=drop)

And then we initialize the tokenizer by instantiating the class
:py:class:`~mindnlp.dataset.transforms.tokenizers.BasicTokenizer`:
::
    from mindnlp.dataset.transforms import BasicTokenizer

    tokenizer = BasicTokenizer(True)

Next, we apply the method :py:class:`process()` to get the processed training set, by passing the
obtained training set, tokenizer, vocabulary and so on into this method:
::
    from mindnlp.dataset import process

    imdb_train = process('imdb', imdb_train, tokenizer=tokenizer, vocab=vocab, \
                        bucket_boundaries=[400, 500], max_len=600, drop_remainder=True)

Finally, we employ the method :py:class:`split()` to split the processed training set, thus getting
a new training set and a validation set:
::
    imdb_train, imdb_valid = imdb_train.split([0.7, 0.3])

Define and Train Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define and Initialize Network
------------------------------------

We introduce the pre-defined network layers from MindSpore and MindNLP to construct
our network.

Use :py:class:`~mindspore.nn.LSTM`, :py:class:`~mindspore.nn.Dense`,
:py:class:`~mindspore.nn.Sigmoid`, :py:class:`~mindspore.nn.Dropout` of MindSpore,
and :py:class:`~mindnlp.abc.Seq2vecModel`,
:py:class:`~mindnlp.modules.encoder.rnn_encoder.RNNEncoder` of MindNLP to construct
our model. And apply :py:class:`~mindspore.common.initializer.Uniform` and
:py:class:`~mindspore.common.initializer.HeUniform` to initialize the weight
and bias of the network we construct.

The code of defining and initializing the network is as follows:
::
    from mindspore import nn
    from mindspore import ops
    from mindspore.common.initializer import Uniform, HeUniform

    from mindnlp.modules import RNNEncoder
    from mindnlp.abc import Seq2vecModel

    class Head(nn.Cell):
        """
        Head for Sentiment Classification model
        """
        def __init__(self, hidden_dim, output_dim, dropout):
            super().__init__()
            weight_init = HeUniform(math.sqrt(5))
            bias_init = Uniform(1 / math.sqrt(hidden_dim * 2))
            self.fc = nn.Dense(hidden_dim * 2, output_dim, weight_init=weight_init, bias_init=bias_init)
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(1 - dropout)

        def construct(self, context):
            context = ops.concat((context[-2, :, :], context[-1, :, :]), axis=1)
            context = self.dropout(context)
            return self.sigmoid(self.fc(context))

    class SentimentClassification(Seq2vecModel):
        """
        Sentiment Classification model
        """
        def __init__(self, encoder, head):
            super().__init__(encoder, head)
            self.encoder = encoder
            self.head = head

        def construct(self, text):
            _, (hidden, _), _ = self.encoder(text)
            output = self.head(hidden)
            return output

    hidden_size = 256
    output_size = 1
    num_layers = 2
    bidirectional = True
    drop = 0.5

    lstm_layer = nn.LSTM(100, hidden_size, num_layers=num_layers, batch_first=True,
                        dropout=drop, bidirectional=bidirectional)

    sentiment_encoder = RNNEncoder(embedding, lstm_layer)

    sentiment_head = Head(hidden_size, output_size, drop)

    net = SentimentClassification(sentiment_encoder, sentiment_head)

Define Loss Function
------------------------------------

A loss function is needed when we train the model. We use :py:class:`~mindspore.nn.BCELoss`
provided by MindSpore to define a loss function:
::
    loss = nn.BCELoss(reduction='mean')

Define Optimizer
------------------------------------

Define the optimizer required for running the model by calling :py:class:`~mindspore.nn.Adam`
and passing the trainable parameters of the model into it:
::
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)

Define Metric
------------------------------------

It is necessary to evaluate the model using one or more metrics. We use
:py:class:`~mindnlp.engine.metrics.accuracy.Accuracy` to define
the metric of the model:
::
    from mindnlp.engine.metrics import Accuracy

    metric = Accuracy()

Train and Evaluate Model
------------------------------------

After defining the network, the loss function, the optimizer and the metric, we employ
:py:class:`~mindnlp.engine.trainer.Trainer` to train and evaluate the model defined above.

The code of training and evaluating the model is as follows:
::
    from mindnlp.engine.trainer import Trainer

    trainer = Trainer(network=net, train_dataset=imdb_train, eval_dataset=imdb_valid, metrics=metric,
                        epochs=5, loss_fn=loss, optimizer=optimizer)

    trainer.run(tgt_columns="label", jit=False)
