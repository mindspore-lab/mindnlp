First Model
===================

Overview
-------------------

Sentiment classification is a classic task in natural language processing.
It is a typical classification problem to mine and analyze people's sentiments
expressed in the text, which is positive or negative. The following uses
MindNLP to implement an RNN-based sentimental classification model to achieve
the following effects:

::

    Input: This film is terrible
    Correct label: Negative
    Forecast label: Negative

    Input: This film is great
    Correct label: Positive
    Forecast label: Positive

Model Building
-------------------

According to the task, the base module :py:class:`~mindnlp.abc.Seq2vecModel`
can be used to build the model. The function of module
:py:class:`~mindnlp.abc.Seq2vecModel` is to extract semantic feature of
the input sequential and calculate to the result vector. It consists of
two modules, ``encoder`` and ``head``, where ``encoder`` mapping the input
sentence into semantic vector and ``head`` performing further calculation
on ``encoder`` output to get the final result.

.. code-block::

    from mindnlp.abc import Seq2vecModel

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

Model Instantiation
-------------------

Two modules ``encoder`` and ``head`` are initialized separately, passing
as arguments into model. We use :py:class:`~mindnlp.modules.RNNEncoder`
provided by MindNLP as the model's ``encoder``, and use customized modules
as the model's ``head``.

.. code-block::

    import math
    from mindspore import nn
    from mindspore import ops
    from mindspore.common.initializer import Uniform, HeUniform
    from mindnlp.modules import Glove
    from mindnlp.modules import RNNEncoder

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

    hidden_size = 256
    output_size = 1
    num_layers = 2
    bidirectional = True
    drop = 0.5
    lr = 0.001

    embedding, vocab = Glove.from_pretrained('6B', 100, special_tokens=["<unk>", "<pad>"], dropout=drop)
    lstm_layer = nn.LSTM(100, hidden_size, num_layers=num_layers, batch_first=True,
                        dropout=drop, bidirectional=bidirectional)
    sentiment_encoder = RNNEncoder(embedding, lstm_layer)
    sentiment_head = Head(hidden_size, output_size, drop)
    net = SentimentClassification(sentiment_encoder, sentiment_head)
