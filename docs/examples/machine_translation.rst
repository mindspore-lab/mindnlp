Machine Translation
===================

`GitHub <https://github.com/mindspore-lab/mindnlp/blob/master/examples/machine_translation.py>`__

Machine translation is the translation of one language (a sentence or a
paragraph or a text) into another language. The following is a demo of
training machine translation using the Multi30k dataset and the Seq2Seq
model.

Define Model
------------

Machine translation is a typical Seq2Seq model that generates another
sequence from one sequence. It involves two processes: one is to
understand the previous sequence and the other is to use the understood
content to generate a new sequence. As for the sequences the model used
can be RNN, LSTM, GRU, other sequence models, etc.

.. code:: python

    from mindnlp.abc import Seq2seqModel

    class MachineTranslation(Seq2seqModel):
        def __init__(self, encoder, decoder):
            super().__init__(encoder, decoder)
            self.encoder = encoder
            self.decoder = decoder

        def construct(self, en, de):
            encoder_out = self.encoder(en)
            decoder_out = self.decoder(de, encoder_out=encoder_out)
            output = decoder_out[0]
            return output.swapaxes(1,2)

Define Hyperparameters
----------------------

The following are some of the required hyperparameters in the model
training process.

.. code:: python

    enc_emb_dim = 256
    dec_emb_dim = 256
    enc_hid_dim = 512
    dec_hid_dim = 512
    enc_dropout = 0.5
    dec_dropout = 0.5

Data Preprocessing
------------------

The dataset was downloaded and preprocessed by calling the interface of
dataset in mindnlp.

Load datasets:

.. code:: python

    from mindnlp.dataset import load

    multi30k_train, multi30k_valid, multi30k_test = load('multi30k')

Initialize the vocab and process the data set:

.. code:: python

    from mindnlp.dataset.transforms import BasicTokenizer
    from mindspore.dataset import text
    from mindnlp.dataset import process

    tokenizer = BasicTokenizer(True) # Tokenizer
    multi30k_train = multi30k_train.map([tokenizer], 'en')
    multi30k_train = multi30k_train.map([tokenizer], 'de')
    en_vocab = text.Vocab.from_dataset(multi30k_train, 'en', special_tokens=['<pad>', '<unk>'], special_first= True) # en
    de_vocab = text.Vocab.from_dataset(multi30k_train, 'de', special_tokens=['<pad>', '<unk>'], special_first= True) # de
    vocab = {'en':en_vocab, 'de':de_vocab}

    multi30k_train = process('multi30k', multi30k_train, vocab=vocab, batch_size=64, max_len = 32, drop_remainder = False)

    multi30k_valid = multi30k_valid.map([tokenizer], 'en')
    multi30k_valid = multi30k_valid.map([tokenizer], 'de')
    multi30k_valid = process('multi30k', multi30k_valid, vocab=vocab, batch_size=64, max_len = 32, drop_remainder = False)

Instantiate Model
-----------------

.. code:: python

    from mindspore import nn
    from mindnlp.modules import RNNEncoder, RNNDecoder

    input_dim = len(en_vocab.vocab())
    output_dim = len(de_vocab.vocab())

    # encoder
    en_embedding = nn.Embedding(input_dim, enc_emb_dim)
    en_rnn = nn.RNN(enc_emb_dim, hidden_size=enc_hid_dim, num_layers=2, has_bias=True,
                    batch_first=True, dropout=enc_dropout, bidirectional=False)
    rnn_encoder = RNNEncoder(en_embedding, en_rnn)

    # decoder
    de_embedding = nn.Embedding(output_dim, dec_emb_dim)
    input_feed_size = 0 if enc_hid_dim == 0 else dec_hid_dim
    rnns = [
        nn.RNNCell(
            input_size=dec_emb_dim + input_feed_size
            if layer == 0
                else dec_hid_dim,
            hidden_size=dec_hid_dim
            )
            for layer in range(2)
    ]
    rnn_decoder = RNNDecoder(de_embedding, rnns, dropout_in=enc_dropout, dropout_out = dec_dropout,attention=True, encoder_output_units=enc_hid_dim)

    net = MachineTranslation(rnn_encoder, rnn_decoder)
    net.update_parameters_name('net.')

Define Optimizer, Loss, Callbacks, Metrics
------------------------------------------

.. code:: python

    from mindnlp.engine.callbacks.timer_callback import TimerCallback
    from mindnlp.engine.callbacks.earlystop_callback import EarlyStopCallback
    from mindnlp.engine.callbacks.best_model_callback import BestModelCallback
    from mindnlp.engine.metrics import Accuracy

    optimizer = nn.Adam(net.trainable_params(), learning_rate=10e-5)
    loss_fn = nn.CrossEntropyLoss()

    # define callbacks
    timer_callback_epochs = TimerCallback(print_steps=-1)
    earlystop_callback = EarlyStopCallback(patience=2)
    bestmodel_callback = BestModelCallback()
    callbacks = [timer_callback_epochs, earlystop_callback, bestmodel_callback]

    # define metrics
    metric = Accuracy()

Define Trainer
--------------

.. code:: python

    from mindnlp.engine.trainer import Trainer

    trainer = Trainer(network=net, train_dataset=multi30k_train, eval_dataset=multi30k_valid, metrics=metric,
                      epochs=10, loss_fn=loss_fn, optimizer=optimizer)

Training Process
----------------

.. code:: python

    trainer.run(tgt_columns="de", jit=True)
    print("end train")


.. parsed-literal::

    Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 454/454 [05:39<00:00,  1.34it/s, loss=3.2271016]
    Evaluate: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:10<00:00,  1.49it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.6223496055226825}


.. parsed-literal::

    Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 454/454 [01:28<00:00,  5.13it/s, loss=2.1794753]
    Evaluate: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:10<00:00,  1.50it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.6646942800788954}


.. parsed-literal::

    Epoch 2: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 454/454 [01:28<00:00,  5.12it/s, loss=1.8816497]
    Evaluate: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:11<00:00,  1.39it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.6863597140039448}


.. parsed-literal::

    Epoch 3: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 454/454 [01:28<00:00,  5.11it/s, loss=1.6710395]
    Evaluate: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:11<00:00,  1.39it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.7070081360946746}


.. parsed-literal::

    Epoch 4: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 454/454 [01:29<00:00,  5.10it/s, loss=1.5266166]
    Evaluate: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:11<00:00,  1.39it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.7174248027613412}


.. parsed-literal::

    Epoch 5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 454/454 [01:29<00:00,  5.10it/s, loss=1.4266685]
    Evaluate: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:11<00:00,  1.38it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.7320019723865878}


.. parsed-literal::

    Epoch 6: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 454/454 [01:29<00:00,  5.09it/s, loss=1.3493056]
    Evaluate: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:11<00:00,  1.37it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.7478427021696252}


.. parsed-literal::

    Epoch 7: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 454/454 [01:29<00:00,  5.09it/s, loss=1.2893807]
    Evaluate: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:11<00:00,  1.38it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.766857741617357}


.. parsed-literal::

    Epoch 8: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 454/454 [01:29<00:00,  5.09it/s, loss=1.2387483]
    Evaluate: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:11<00:00,  1.40it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.777120315581854}


.. parsed-literal::

    Epoch 9: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 454/454 [01:29<00:00,  5.09it/s, loss=1.1957376]
    Evaluate: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:11<00:00,  1.38it/s]

.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.782482741617357}
    end train


