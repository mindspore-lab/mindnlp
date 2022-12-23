机器翻译
========

机器翻译（Machine
Translation）就是将一种语言（一句话或者一段话或者一篇文章）翻译成另外一种语言。

这里我们使用mindnlp框架，multi30k数据集为例，搭建神经网络模型

步骤
----

加载Multi30k数据集
~~~~~~~~~~~~~~~~~~

mindnlp框架中的dataset部分提供了NLP不同领域的数据集加载，在这里我们使用multi30k数据集。

若数据集未下载则会自动下载，已下载数据集可以指定路径加载。

.. code:: ipython3

    from mindnlp.dataset import load

    multi30k_train, multi30k_valid, multi30k_test = load('multi30k')

预处理数据集
~~~~~~~~~~~~

mindnlp框架里面对于已经提供的数据集都有对应的\ ``process``\ 函数。

这里我们使用\ ``text.Vocab.from_dataset``\ 通过multi30k数据集的训练集获得\ ``vocab``\ 后，使用\ ``process``\ 函数进行处理。

.. code:: ipython3

    from mindnlp.dataset.transforms import BasicTokenizer
    from mindspore.dataset import text
    from mindnlp.dataset import process

    tokenizer = BasicTokenizer(True) # Tokenizer
    multi30k_train = multi30k_train.map([tokenizer], 'en')
    multi30k_train = multi30k_train.map([tokenizer], 'de')
    en_vocab = text.Vocab.from_dataset(multi30k_train, 'en', special_tokens=['<pad>', '<unk>'], special_first= True) # 获得en词表
    de_vocab = text.Vocab.from_dataset(multi30k_train, 'de', special_tokens=['<pad>', '<unk>'], special_first= True) # 获得de词表
    vocab = {'en':en_vocab, 'de':de_vocab}

    multi30k_train = process('multi30k', multi30k_train, vocab=vocab, batch_size=64, max_len = 32, drop_remainder = False) # 对训练集进行预处理

    multi30k_valid = multi30k_valid.map([tokenizer], 'en')
    multi30k_valid = multi30k_valid.map([tokenizer], 'de')
    multi30k_valid = process('multi30k', multi30k_valid, vocab=vocab, batch_size=64, max_len = 32, drop_remainder = False) # 对验证集进行预处理

创建模型
~~~~~~~~

定义基本参数并基于RNN搭建Seq2Seq模型，其中包含了一个编码器（encoder）和一个解码器（decoder）。

.. code:: ipython3

    from mindspore import nn
    from mindnlp.abc import Seq2seqModel
    from mindnlp.modules import RNNEncoder, RNNDecoder

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

    # define Models & Loss & Optimizer
    input_dim = len(en_vocab.vocab())
    output_dim = len(de_vocab.vocab())
    enc_emb_dim = 256
    dec_emb_dim = 256
    enc_hid_dim = 512
    dec_hid_dim = 512
    enc_dropout = 0.5
    dec_dropout = 0.5

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


训练模型
~~~~~~~~

使用\ ``trainer``\ 可以让我们十分方便的训练模型，并打印训练过程的相关信息。

.. code:: ipython3

    from mindnlp.engine.callbacks.timer_callback import TimerCallback
    from mindnlp.engine.callbacks.earlystop_callback import EarlyStopCallback
    from mindnlp.engine.callbacks.best_model_callback import BestModelCallback
    from mindnlp.engine.metrics import Accuracy
    from mindnlp.engine.trainer import Trainer

    optimizer = nn.Adam(net.trainable_params(), learning_rate=10e-5)
    loss_fn = nn.CrossEntropyLoss()

    # define callbacks
    timer_callback_epochs = TimerCallback(print_steps=-1)
    earlystop_callback = EarlyStopCallback(patience=2)
    bestmodel_callback = BestModelCallback()
    callbacks = [timer_callback_epochs, earlystop_callback, bestmodel_callback]

    # define metrics
    metric = Accuracy()

    # define trainer
    trainer = Trainer(network=net, train_dataset=multi30k_train, eval_dataset=multi30k_valid, metrics=metric,
                      epochs=10, loss_fn=loss_fn, optimizer=optimizer)
    trainer.run(tgt_columns="de", jit=True)
    print("end train")



.. parsed-literal::

    Epoch 0: 100%|██████████| 454/454 [01:47<00:00,  4.23it/s, loss=3.3234878]
    Evaluate: 100%|██████████| 16/16 [00:11<00:00,  1.37it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.6108234714003945}


.. parsed-literal::

    Epoch 1: 100%|██████████| 454/454 [01:29<00:00,  5.10it/s, loss=2.2835732]
    Evaluate: 100%|██████████| 16/16 [00:11<00:00,  1.37it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.6642320019723866}


.. parsed-literal::

    Epoch 2: 100%|██████████| 454/454 [01:29<00:00,  5.10it/s, loss=2.035218]
    Evaluate: 100%|██████████| 16/16 [00:11<00:00,  1.38it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.6789632642998028}


.. parsed-literal::

    Epoch 3: 100%|██████████| 454/454 [01:29<00:00,  5.09it/s, loss=1.8333515]
    Evaluate: 100%|██████████| 16/16 [00:11<00:00,  1.37it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.685435157790927}


.. parsed-literal::

    Epoch 4: 100%|██████████| 454/454 [01:29<00:00,  5.09it/s, loss=1.6696163]
    Evaluate: 100%|██████████| 16/16 [00:11<00:00,  1.36it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.7221708579881657}


.. parsed-literal::

    Epoch 5: 100%|██████████| 454/454 [01:29<00:00,  5.09it/s, loss=1.5085986]
    Evaluate: 100%|██████████| 16/16 [00:11<00:00,  1.38it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.7530202169625246}


.. parsed-literal::

    Epoch 6: 100%|██████████| 454/454 [01:28<00:00,  5.10it/s, loss=1.3471557]
    Evaluate: 100%|██████████| 16/16 [00:11<00:00,  1.37it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.7818047337278107}


.. parsed-literal::

    Epoch 7: 100%|██████████| 454/454 [01:29<00:00,  5.09it/s, loss=1.2089082]
    Evaluate: 100%|██████████| 16/16 [00:11<00:00,  1.37it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.8039632642998028}


.. parsed-literal::

    Epoch 8: 100%|██████████| 454/454 [01:29<00:00,  5.09it/s, loss=1.0923911]
    Evaluate: 100%|██████████| 16/16 [00:11<00:00,  1.38it/s]


.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.8340421597633136}


.. parsed-literal::

    Epoch 9: 100%|██████████| 454/454 [01:28<00:00,  5.10it/s, loss=0.9825405]
    Evaluate: 100%|██████████| 16/16 [00:11<00:00,  1.38it/s]

.. parsed-literal::

    Evaluate Score: {'Accuracy': 0.8553377712031558}
    end train


.. parsed-literal::



