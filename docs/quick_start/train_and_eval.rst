Train and Eval
===================

We use the task of sentiment analysis
to give a detailed introduction. And we employ
:py:class:`~mindnlp.engine.trainer.Trainer`
for a fast training and evaluation.

Load and Process Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Load Data
------------------------------------

We can call the function ``load()`` from :py:class:`~mindnlp.dataset`
to load the IMDB dataset. And then the method will return the training set
and the testing set of the IMDB dataset.

The code of loading data is as follows:

.. code-block:: python

    from mindnlp.dataset import load

    imdb_train, imdb_test = load('imdb', shuffle=True)

Process and Split Data
------------------------------------

First we obtain the embeddings and the vocabulary, by calling the function
``from_pretrained()`` from
:py:class:`~mindnlp.modules.embeddings.glove_embedding.Glove`:

.. code-block:: python

    from mindnlp.modules import Glove

    embedding, vocab = Glove.from_pretrained('6B', 100, special_tokens=["<unk>", "<pad>"], dropout=drop)

And then we initialize the tokenizer by instantiating the class
:py:class:`~mindnlp.dataset.transforms.tokenizers.BasicTokenizer`:

.. code-block:: python

    from mindnlp.dataset.transforms import BasicTokenizer

    tokenizer = BasicTokenizer(True)

Next, we apply the method ``process()`` to get the processed
training set, by passing the obtained training set, tokenizer, vocabulary
and so on into this method:

.. code-block:: python

    from mindnlp.dataset import process

    imdb_train = process('imdb', imdb_train, tokenizer=tokenizer, vocab=vocab, \
                        bucket_boundaries=[400, 500], max_len=600, drop_remainder=True)

Finally, we employ the method ``split()`` to split the processed
training set, thus getting a new training set and a validation set:

.. code-block:: python

    imdb_train, imdb_valid = imdb_train.split([0.7, 0.3])

Define and Train Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define and Initialize Network
------------------------------------

We introduce the pre-defined network layers from MindSpore and MindNLP
to construct our network.

Use ``mindspore.nn.LSTM``,
``mindspore.nn.Dense``,
``mindspore.nn.Sigmoid``, ``mindspore.nn.Dropout``
of MindSpore, and :py:class:`~mindnlp.abc.Seq2vecModel`,
:py:class:`~mindnlp.modules.encoder.rnn_encoder.RNNEncoder`
of MindNLP to construct our model. And apply
``mindspore.common.initializer.Uniform`` and
``mindspore.common.initializer.HeUniform`` of MindSpore
to initialize the weight and bias of the network we construct.

The code of defining and initializing the network is as follows:

.. code-block:: python

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

Define Loss Function and Optimizer
------------------------------------

A loss function is needed when we train the model. We use
``mindspore.nn.BCELoss``
provided by MindSpore to define a loss function:

.. code-block:: python

    loss = nn.BCELoss(reduction='mean')

After that, define the optimizer required for running the model by calling
``mindspore.nn.Adam`` and passing the trainable parameters of the model
into it:

.. code-block:: python

    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)

Define Callbacks
------------------------------------

Defining callbacks plays an important role in training models with MindNLP.
It helps to add some extra operations to the training process.

For example, we can add
:py:class:`~mindnlp.engine.callbacks.best_model_callback.BestModelCallback`
for saving and loading the best model. Or we can use
:py:class:`~mindnlp.engine.callbacks.checkpoint_callback.CheckpointCallback`
for saving the checkpoint. There are also other callbacks for early stop
and timing.

When customizing the callbacks we need, we could first initialize
the corresponding classes of callbacks, and then declare a callback list
of the callbacks we initialized before. Just like:

.. code-block:: python

    from mindnlp.engine.callbacks.timer_callback import TimerCallback
    from mindnlp.engine.callbacks.earlystop_callback import EarlyStopCallback
    from mindnlp.engine.callbacks.best_model_callback import BestModelCallback

    timer_callback_epochs = TimerCallback(print_steps=2)
    earlystop_callback = EarlyStopCallback(patience=2)
    bestmodel_callback = BestModelCallback(save_path='save/callback/best_model', auto_load=True)

    callbacks = [timer_callback_epochs, earlystop_callback, bestmodel_callback]

Define Metrics
------------------------------------

It is necessary to evaluate the model using one or more metrics. We choose
:py:class:`~mindnlp.engine.metrics.accuracy.Accuracy` to be
the metric of the model:

.. code-block:: python

    from mindnlp.engine.metrics import Accuracy

    metric = Accuracy()

Train and Evaluate Model
------------------------------------

After defining the network, the loss function, the optimizer, the callbacks
and the metrics, we employ :py:class:`~mindnlp.engine.trainer.Trainer` to
train and evaluate the model defined above.

More specifically, when we train the model, we should pass these parameters
into :py:class:`~mindnlp.engine.trainer.Trainer`:

- ``network``: the network to be trained.
- ``train_dataset``: the dataset for training the model.
- ``eval_dataset``: the dataset for evaluating the model.
- ``metrics``: the metrics used for model evaluation.
- ``epochs``: the total number of training iterations.
- ``loss_fn``: the loss function.
- ``optimizer``: the optimizer for updating the trainable parameters.
- ``callbacks``: the additional operations executed when training.

The example code of training and evaluating the model is as follows:

.. code-block:: python

    from mindnlp.engine.trainer import Trainer

    trainer = Trainer(network=net, train_dataset=imdb_train, eval_dataset=imdb_valid, metrics=metric,
                        epochs=5, loss_fn=loss, optimizer=optimizer, callbacks=callbacks)

    trainer.run(tgt_columns="label", jit=False)
